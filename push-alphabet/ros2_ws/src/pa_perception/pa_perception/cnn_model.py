"""Small CNN for letter class / orientation / center offset (docs/cnn.md). Needs torch.

Input  `input`  float32 [N, 1, 96, 96] (pa_perception.cnn_crop.extract_crop).
Outputs `logits` [N, 4] (I, U, Y, K), `angle` [N, 2] = (cos phi, sin phi) (not
normalized), `offset` [N, 2] in cm.

Seven 3x3 conv + BN + ReLU blocks (four with stride 2: 96 -> 6), a 1x1
bottleneck, then a flatten (not a global pool: the offset head needs where the
letter is) and a shared FC layer feeding the three heads. ~0.35 M parameters.
"""

import torch
from torch import nn

CHANNELS = (16, 32, 32, 64, 64, 96, 96)
STRIDES = (2, 1, 2, 1, 2, 1, 2)


def _block(cin, cout, stride):
    return nn.Sequential(nn.Conv2d(cin, cout, 3, stride, 1, bias=False),
                         nn.BatchNorm2d(cout), nn.ReLU(inplace=True))


class LetterCNN(nn.Module):
    def __init__(self, n_classes=4, channels=CHANNELS, strides=STRIDES, bottleneck=32,
                 hidden=128, dropout=0.1, input_size=96):
        super().__init__()
        layers, cin = [], 1
        for c, s in zip(channels, strides):
            layers.append(_block(cin, c, s))
            cin = c
        layers.append(_block_1x1(cin, bottleneck))
        self.features = nn.Sequential(*layers)
        side = input_size
        for s in strides:
            side = (side + 2 - 3) // s + 1
        self.trunk = nn.Sequential(nn.Flatten(), nn.Linear(bottleneck * side * side, hidden),
                                   nn.ReLU(inplace=True), nn.Dropout(dropout))
        self.cls = nn.Linear(hidden, n_classes)
        self.angle = nn.Linear(hidden, 2)
        self.offset = nn.Linear(hidden, 2)

    def forward(self, x):
        h = self.trunk(self.features(x))
        return self.cls(h), self.angle(h), self.offset(h)


def _block_1x1(cin, cout):
    return nn.Sequential(nn.Conv2d(cin, cout, 1, bias=False), nn.BatchNorm2d(cout),
                         nn.ReLU(inplace=True))


def n_params(model):
    return sum(p.numel() for p in model.parameters())


def export_onnx(model, path, opset=17):
    """Export with input `input`, outputs `logits`, `angle`, `offset`, dynamic batch."""
    model = model.eval()
    dummy = torch.zeros(1, 1, 96, 96)
    kw = dict(input_names=["input"], output_names=["logits", "angle", "offset"],
              dynamic_axes={"input": {0: "N"}, "logits": {0: "N"}, "angle": {0: "N"},
                            "offset": {0: "N"}},
              opset_version=opset, do_constant_folding=True)
    try:
        torch.onnx.export(model, dummy, str(path), dynamo=False, **kw)
    except TypeError:  # older torch without the `dynamo` keyword
        torch.onnx.export(model, dummy, str(path), **kw)
    return path
