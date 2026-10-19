"""
Download the robomimic ph (proficient-human) datasets packaged by Diffusion Policy,
without downloading the full zip archives (robomimic_image.zip is 85 GB).

Only the zip index is read, then each requested file is streamed out of the archive
with an HTTP range request, decompressed on the fly and checked against its CRC.

Usage (from the repo root):
python chunked_policy/scripts/download_robomimic.py                        # all tasks, lowdim + vision, _abs
python chunked_policy/scripts/download_robomimic.py --tasks lift can square --obs vision
python chunked_policy/scripts/download_robomimic.py --dry-run              # only print the files and sizes
"""
import argparse
import io
import os
import pathlib
import shutil
import struct
import urllib.request
import zipfile
import zlib
import tqdm

URL = 'https://diffusion-policy.cs.columbia.edu/data/training/'
ZIPS = {'lowdim': 'robomimic_lowdim.zip', 'vision': 'robomimic_image.zip'}
PREFIX = {'lowdim': 'low_dim', 'vision': 'image'}
TASKS = ['lift', 'can', 'square', 'tool_hang', 'transport']
CHUNK = 1 << 24


class HttpFile(io.RawIOBase):
    """Read-only seekable file over HTTP range requests, enough for zipfile to read the index."""
    def __init__(self, url):
        self.url = url
        self.pos = 0
        req = urllib.request.Request(url, method='HEAD')
        self.size = int(urllib.request.urlopen(req).headers['Content-Length'])

    def seekable(self):
        return True

    def readable(self):
        return True

    def tell(self):
        return self.pos

    def seek(self, offset, whence=0):
        self.pos = {0: offset, 1: self.pos + offset, 2: self.size + offset}[whence]
        return self.pos

    def readinto(self, b):
        if self.pos >= self.size:
            return 0
        end = min(self.pos + len(b), self.size) - 1
        data = read_range(self.url, self.pos, end)
        b[:len(data)] = data
        self.pos += len(data)
        return len(data)


def read_range(url, start, end):
    req = urllib.request.Request(url, headers={'Range': f'bytes={start}-{end}'})
    return urllib.request.urlopen(req).read()


def extract(url, info: zipfile.ZipInfo, out_path: pathlib.Path):
    # the data starts after the local file header (30 bytes + file name + extra field)
    header = read_range(url, info.header_offset, info.header_offset + 29)
    name_len, extra_len = struct.unpack('<HH', header[26:30])
    start = info.header_offset + 30 + name_len + extra_len
    end = start + info.compress_size - 1

    assert info.compress_type in (zipfile.ZIP_STORED, zipfile.ZIP_DEFLATED)
    inflate = zlib.decompressobj(-15) if info.compress_type == zipfile.ZIP_DEFLATED else None
    crc = 0
    tmp_path = out_path.with_name(out_path.name + '.part')
    req = urllib.request.Request(url, headers={'Range': f'bytes={start}-{end}'})
    with urllib.request.urlopen(req) as response, open(tmp_path, 'wb') as f, \
            tqdm.tqdm(total=info.file_size, unit='B', unit_scale=True, desc=out_path.name) as pbar:
        while True:
            chunk = response.read(CHUNK)
            if not chunk:
                break
            data = inflate.decompress(chunk) if inflate else chunk
            crc = zlib.crc32(data, crc)
            f.write(data)
            pbar.update(len(data))
        if inflate:
            data = inflate.flush()
            crc = zlib.crc32(data, crc)
            f.write(data)
    if crc != info.CRC or tmp_path.stat().st_size != info.file_size:
        tmp_path.unlink()
        raise RuntimeError(f'{info.filename}: corrupted download (CRC or size mismatch)')
    tmp_path.rename(out_path)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--tasks', nargs='+', default=TASKS, choices=TASKS)
    parser.add_argument('--obs', nargs='+', default=['lowdim', 'vision'], choices=['lowdim', 'vision'])
    parser.add_argument('--actions', nargs='+', default=['abs'], choices=['abs', 'delta'],
        help='abs: <obs>_abs.hdf5 (used by the *_abs task configs, paper setup), delta: <obs>.hdf5')
    parser.add_argument('--out', default='data', help='the configs expect data/robomimic/datasets/<task>/ph/')
    parser.add_argument('--dry-run', action='store_true')
    args = parser.parse_args()

    # select the files in the zip indexes
    todo = list()
    for obs in args.obs:
        url = URL + ZIPS[obs]
        index = {i.filename: i for i in zipfile.ZipFile(io.BufferedReader(HttpFile(url), 1 << 20)).infolist()}
        for task in args.tasks:
            for actions in args.actions:
                name = f'robomimic/datasets/{task}/ph/{PREFIX[obs]}{"_abs" if actions == "abs" else ""}.hdf5'
                out_path = pathlib.Path(args.out) / name
                if out_path.exists() and out_path.stat().st_size == index[name].file_size:
                    print(f'already there: {out_path}')
                    continue
                todo.append((url, index[name], out_path))

    total = sum(info.file_size for _, info, _ in todo)
    for _, info, out_path in todo:
        print(f'{info.file_size / 1e9:7.2f} GB  {out_path}')
    print(f'{total / 1e9:7.2f} GB  total to download')
    if args.dry_run or not todo:
        return

    pathlib.Path(args.out).mkdir(parents=True, exist_ok=True)
    free = shutil.disk_usage(args.out).free
    if total > free:
        raise SystemExit(f'not enough disk space: {total / 1e9:.1f} GB needed, {free / 1e9:.1f} GB free')

    for url, info, out_path in todo:
        out_path.parent.mkdir(parents=True, exist_ok=True)
        extract(url, info, out_path)


if __name__ == '__main__':
    main()
