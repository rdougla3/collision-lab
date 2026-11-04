"""Network construction is independent of data collection and training."""
from dataclasses import asdict, dataclass, field

import torch
from torch import nn
from torch.nn.utils.rnn import pack_padded_sequence


@dataclass
class NetworkSpec:
    kind: str = "feedforward"
    input_size: int = 6
    output_size: int = 1
    hidden_sizes: list[int] = field(default_factory=lambda: [32, 16])
    hidden_size: int = 32
    num_layers: int = 1

    def validate(self):
        if self.kind not in ("feedforward", "rnn", "gru"):
            raise ValueError("Network kind must be feedforward, rnn, or gru")
        for name, value in [("input_size", self.input_size), ("output_size", self.output_size),
                            ("hidden_size", self.hidden_size), ("num_layers", self.num_layers)]:
            if type(value) is not int or value < 1:
                raise ValueError(f"{name} must be a positive integer")
        if not isinstance(self.hidden_sizes, list) or not self.hidden_sizes or any(type(n) is not int or n < 1 for n in self.hidden_sizes):
            raise ValueError("hidden_sizes must be a nonempty list of positive integers")
        if max(self.hidden_sizes + [self.hidden_size]) > 512 or self.num_layers > 4 or len(self.hidden_sizes) > 4:
            raise ValueError("Browser models support up to four hidden layers of at most 512 units")


class CollisionNetwork(nn.Module):
    def __init__(self, spec: NetworkSpec):
        super().__init__()
        spec.validate()
        self.spec = spec
        if spec.kind == "feedforward":
            sizes = [spec.input_size, *spec.hidden_sizes]
            self.layers = nn.ModuleList(nn.Linear(a, b) for a, b in zip(sizes, sizes[1:]))
            self.output = nn.Linear(sizes[-1], spec.output_size)
        else:
            recurrent = nn.GRU if spec.kind == "gru" else nn.RNN
            self.recurrent = recurrent(spec.input_size, spec.hidden_size,
                                       num_layers=spec.num_layers, batch_first=True)
            self.output = nn.Linear(spec.hidden_size, spec.output_size)

    def forward(self, inputs, lengths=None):
        """Return logits; inputs are [batch, time, features], right padded."""
        if inputs.ndim == 2:
            inputs = inputs.unsqueeze(1)
        if lengths is None:
            lengths = torch.full((inputs.size(0),), inputs.size(1), dtype=torch.long)
        if self.spec.kind == "feedforward":
            hidden = inputs[torch.arange(inputs.size(0), device=inputs.device), lengths.to(inputs.device) - 1]
            for layer in self.layers:
                hidden = torch.relu(layer(hidden))
        else:
            packed = pack_padded_sequence(inputs, lengths.cpu(), batch_first=True, enforce_sorted=False)
            _, hidden = self.recurrent(packed)
            hidden = hidden[-1]
        return self.output(hidden)

    def export_weights(self):
        return {name: tensor.detach().cpu().tolist() for name, tensor in self.state_dict().items()}


def make_network(kind="feedforward", **dimensions):
    """Accept a NetworkSpec or a type plus dimension parameters."""
    spec = kind if isinstance(kind, NetworkSpec) else NetworkSpec(kind=kind, **dimensions)
    return CollisionNetwork(NetworkSpec(**asdict(spec)))
