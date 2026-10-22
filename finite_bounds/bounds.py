import math
import torch


def bound_margin(net, target, rival, lower, upper, alpha, steps=500, lr=1.0, decay=0.95):
    """Return (lower bound, feasible upper bound, witness) for one class margin.
    """
    l, u = lower.detach().reshape(1, -1), upper.detach().reshape(1, -1)
    if l.dtype != torch.float64 or u.dtype != l.dtype or l.device != u.device:
        raise ValueError("Box endpoints must be float64 on the same device")
    if l.shape != u.shape or not l.numel() or not bool(
            torch.isfinite(l).all() and torch.isfinite(u).all() and (l <= u).all()):
        raise ValueError("Expected one finite, ordered box")
    if not isinstance(steps, int) or steps < 0 or not math.isfinite(lr) or lr <= 0:
        raise ValueError("steps must be nonnegative and lr positive")
    if not math.isfinite(decay) or not 0 < decay <= 1:
        raise ValueError("decay must be in (0, 1]")
    a = torch.as_tensor(alpha, dtype=l.dtype, device=l.device).detach().reshape(1, -1)
    if a.numel() not in (1, l.numel()) or not bool(torch.isfinite(a).all() and (a >= 0).all()):
        raise ValueError("alpha must be a finite nonnegative scalar or coordinate vector")
    x = l / 2 + u / 2
    best_lower, best_upper, witness = -math.inf, math.inf, None
    for step in range(steps + 1):
        with torch.enable_grad():
            x = x.detach().requires_grad_(True)
            scores = net(x)
            margin = scores[0, target] - scores[0, rival]
            h = margin + (a * (x - l) * (x - u)).sum()
            grad, = torch.autograd.grad(h, x)
        x, grad = x.detach(), grad.detach()
        tangent = h.detach() + torch.minimum(grad * (l - x), grad * (u - x)).sum()
        if not bool(torch.isfinite(tangent) and torch.isfinite(margin) and torch.isfinite(grad).all()):
            raise FloatingPointError("Nonfinite bound evaluation")
        best_lower = max(best_lower, float(tangent))
        if float(margin.detach()) < best_upper:
            best_upper, witness = float(margin.detach()), x.clone()
        if best_lower > best_upper:
            raise FloatingPointError("Inconsistent lower and upper bounds")
        if step < steps:
            x = torch.maximum(l, torch.minimum(u, x - lr * grad))
            lr *= decay
    return best_lower, best_upper, witness
