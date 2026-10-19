import torch.nn as nn


class MLP(nn.Sequential):
    """Linear -> ReLU (-> Dropout) for each hidden layer, then a final Linear."""
    def __init__(self, in_dim, hidden_dims, out_dim, dropout=0.0):
        layers = list()
        for hidden_dim in hidden_dims:
            layers.append(nn.Linear(in_dim, hidden_dim))
            layers.append(nn.ReLU())
            if dropout > 0:
                layers.append(nn.Dropout(dropout))
            in_dim = hidden_dim
        layers.append(nn.Linear(in_dim, out_dim))
        super().__init__(*layers)
