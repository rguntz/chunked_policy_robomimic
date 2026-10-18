from typing import Dict
from diffusion_policy.policy.base_vision_policy import BaseVisionPolicy

class BaseVisionRunner:
    def __init__(self, output_dir):
        self.output_dir = output_dir

    def run(self, policy: BaseVisionPolicy) -> Dict:
        raise NotImplementedError()
