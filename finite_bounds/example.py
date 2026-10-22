"""Quadratic CCP"""
import torch
from torch import nn

from . import CurvatureBounds, bound_margin


class QuadraticCCP(nn.Module):
    def __init__(self):
        super().__init__()
        self.n_degree, self.weight_sharing = 2, False
        self.U1 = nn.Linear(1, 1, bias=False)
        self.U2 = nn.Linear(1, 1, bias=False)
        self.C = nn.Linear(1, 2)
        with torch.no_grad():
            self.U1.weight.fill_(1)
            self.U2.weight.fill_(-1)
            self.C.weight.copy_(torch.tensor([[1.], [0.]]))
            self.C.bias.copy_(torch.tensor([0.75, 0.]))

    def forward(self, x):
        return self.C(self.U1(x) * (1 + self.U2(x)))


def main():
    net = QuadraticCCP().double().eval().requires_grad_(False)
    lower = torch.tensor([[-1.]], dtype=torch.float64)
    upper = -lower
    alpha = CurvatureBounds(net).alpha(0, 1)
    L, U, witness = bound_margin(net, 0, 1, lower, upper, alpha, steps=10)
    assert alpha >= 1.0
    assert L <= -1.25 <= U
    assert torch.all((lower <= witness) & (witness <= upper))
    print(f"alpha={alpha:.12g}, lower={L:.12g}, upper={U:.12g}")


if __name__ == "__main__":
    main()
