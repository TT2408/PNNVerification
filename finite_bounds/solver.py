"""Finite PGD and BaB used by the MNIST finite-computation experiments.

Alpha must make the convexified margin convex on the entire box. Network
values and gradients use float64, not directed interval arithmetic.
"""
from __future__ import annotations

import heapq
import math
import time
from typing import Any, Iterable

import torch


def _box(l: torch.Tensor, u: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    l, u = l.detach(), u.detach()
    if l.ndim == 1:
        l = l.unsqueeze(0)
    if u.ndim == 1:
        u = u.unsqueeze(0)
    if l.ndim != 2 or l.shape[0] != 1 or l.shape != u.shape or l.numel() == 0:
        raise ValueError("l and u must describe one nonempty box, shape (1, d) or (d,)")
    if l.device != u.device or l.dtype != u.dtype or not l.is_floating_point():
        raise ValueError("box endpoints must have the same floating dtype and device")
    if not bool(torch.isfinite(l).all() and torch.isfinite(u).all() and (l <= u).all()):
        raise ValueError("box endpoints must be finite and ordered")
    return l, u


def optimize_box(
    net: Any,
    tc: int,
    ac: int,
    l: torch.Tensor,
    u: torch.Tensor,
    alpha: Any,
    *,
    max_steps: int = 100,
    lr: float = 1.0,
    gamma: float = 0.95,
    deadline: float | None = None,
    checkpoints: Iterable[int] | None = None,
    start: torch.Tensor | None = None,
    safety_margin: float = 0.0,
    certify: bool = True,
) -> dict[str, Any]:
    begun = time.monotonic()
    l, u = _box(l, u)
    if max_steps < 0 or int(max_steps) != max_steps:
        raise ValueError("max_steps must be a nonnegative integer")
    if not math.isfinite(lr) or lr <= 0 or not math.isfinite(gamma) or not 0 < gamma <= 1:
        raise ValueError("lr must be positive and 0 < gamma <= 1")
    if not math.isfinite(safety_margin) or safety_margin < 0:
        raise ValueError("safety_margin must be finite and nonnegative")
    alpha = torch.as_tensor(alpha if certify else 0.0, dtype=l.dtype, device=l.device).detach()
    if alpha.numel() not in (1, l.numel()):
        raise ValueError("alpha must be scalar or have one entry per input coordinate")
    alpha = alpha.reshape(1, -1)
    result: dict[str, Any] = dict(
        L=-math.inf, U=math.inf, z=None, raw=math.nan, raw_min=math.inf,
        gap=math.nan, L_iter=-math.inf, steps=0, evaluations=0, trace=[],
        reason="max_steps", finite=True, elapsed=0.0, certify=certify,
    )
    if not bool(torch.isfinite(alpha).all() and (alpha >= 0).all()):
        result.update(finite=False, reason="invalid_alpha", elapsed=time.monotonic() - begun)
        return result
    x = l / 2 + u / 2 if start is None else start.detach().to(l).reshape(l.shape)
    x = torch.maximum(l, torch.minimum(u, x)).detach()
    wanted = None if checkpoints is None else set(int(k) for k in checkpoints)
    last_record = None
    current_lr = lr
    for step in range(int(max_steps) + 1):
        if deadline is not None and time.monotonic() >= deadline:
            result["reason"] = "timeout"
            break
        with torch.enable_grad():
            x = x.detach().requires_grad_(True)
            output = net(x)
            original = output[0, tc] - output[0, ac]
            h = original + (alpha * (x - l) * (x - u)).sum()
            grad, = torch.autograd.grad(h, x, create_graph=False, retain_graph=False)
        x, grad, h, original = x.detach(), grad.detach(), h.detach(), original.detach()
        if not bool(torch.isfinite(h) and torch.isfinite(original) and torch.isfinite(grad).all()):
            result.update(L=-math.inf, finite=False, reason="nonfinite")
            break
        gap_tensor = -torch.minimum(grad * (l - x), grad * (u - x)).sum()
        gap = float(gap_tensor.item())
        raw, f = float(h.item()), float(original.item())
        tangent = float((h - gap_tensor).item()) - safety_margin
        if not all(math.isfinite(v) for v in (gap, raw, f, tangent)):
            result.update(L=-math.inf, finite=False, reason="nonfinite")
            break
        lower = tangent if certify else -math.inf
        result["L"] = max(result["L"], lower)
        if f < result["U"]:
            result["U"], result["z"] = f, x.clone()
        result.update(raw=raw, gap=gap, L_iter=lower, steps=step,
                      evaluations=result["evaluations"] + 1,
                      raw_min=min(result["raw_min"], raw))
        last_record = dict(step=step, raw=raw, gap=gap, L_iter=lower,
                           L=result["L"], U=result["U"], f=f,
                           elapsed=time.monotonic() - begun)
        if wanted is None or step in wanted or step == max_steps:
            result["trace"].append(last_record)
        if certify and result["L"] > result["U"]:
            result.update(L=-math.inf, finite=False, reason="inconsistent_bounds")
            break
        if step < max_steps:
            x = torch.maximum(l, torch.minimum(u, x - current_lr * grad)).detach()
            current_lr *= gamma
    if last_record is not None and (not result["trace"] or result["trace"][-1]["step"] != last_record["step"]):
        result["trace"].append(last_record)
    result["elapsed"] = time.monotonic() - begun
    return result


def _status(lower: float, upper: float, finite: bool, margin: float) -> str:
    if not finite or math.isnan(lower) or math.isnan(upper) or lower > upper:
        return "UNKNOWN"
    if upper < -margin:
        return "ADVERSARIAL"
    if math.isfinite(lower) and lower > 0:
        return "VERIFIED"
    return "UNKNOWN"


def solve_property(
    net: Any,
    tc: int,
    ac: int,
    l: torch.Tensor,
    u: torch.Tensor,
    alpha: Any,
    *,
    max_steps: int = 100,
    lr: float = 1.0,
    gamma: float = 0.95,
    deadline: float | None = None,
    timeout: float | None = None,
    root_only: bool = False,
    max_nodes: int | None = None,
    max_splits: int | None = None,
    n_branches: int = 2,
    checkpoints: Iterable[int] | None = None,
    safety_margin: float = 0.0,
    upper_steps: int | None = None,
    upper_lr: float | None = None,
) -> dict[str, Any]:
    begun = time.monotonic()
    l, u = _box(l, u)
    if timeout is not None:
        if not math.isfinite(timeout) or timeout < 0:
            raise ValueError("timeout must be finite and nonnegative")
        deadline = min(deadline, begun + timeout) if deadline is not None else begun + timeout
    if max_nodes is not None:
        if max_splits is not None and max_splits != max_nodes:
            raise ValueError("max_nodes and max_splits must agree if both are given")
        max_splits = max_nodes
    if max_splits is not None and (max_splits < 0 or int(max_splits) != max_splits):
        raise ValueError("max_splits must be a nonnegative integer or None")
    if n_branches < 2 or int(n_branches) != n_branches:
        raise ValueError("n_branches must be an integer >= 2")
    upper_steps = max_steps if upper_steps is None else upper_steps
    upper_lr = (100.0 if hasattr(net, "activation") else 1.0) if upper_lr is None else upper_lr
    if upper_steps < 0 or int(upper_steps) != upper_steps:
        raise ValueError("upper_steps must be a nonnegative integer")
    if not math.isfinite(upper_lr) or upper_lr <= 0:
        raise ValueError("upper_lr must be finite and positive")
    kwargs = dict(max_steps=max_steps, lr=lr, gamma=gamma, deadline=deadline,
                  safety_margin=safety_margin)

    attack_totals = dict(attack_steps=0, attack_evaluations=0, attack_elapsed=0.0)

    def improve_upper(node_result, node_l, node_u):
        if not node_result["finite"]:
            return
        attack = optimize_box(net, tc, ac, node_l, node_u, 0.0, certify=False,
                              max_steps=upper_steps, lr=upper_lr, gamma=gamma,
                              deadline=deadline, checkpoints=())
        node_result["upper_attack"] = attack
        attack_totals["attack_steps"] += attack["steps"]
        attack_totals["attack_evaluations"] += attack["evaluations"]
        attack_totals["attack_elapsed"] += attack["elapsed"]
        if math.isfinite(attack["U"]) and attack["z"] is not None and attack["U"] < node_result["U"]:
            node_result["U"], node_result["z"] = attack["U"], attack["z"]
        if node_result["L"] > node_result["U"]:
            node_result.update(finite=False, reason="inconsistent_bounds")

    root = optimize_box(net, tc, ac, l, u, alpha, checkpoints=checkpoints, **kwargs)
    improve_upper(root, l, u)
    lower, upper, witness = root["L"], root["U"], root["z"]
    finite = root["finite"]
    splits, nodes, attempted_nodes = 0, 1, 1
    leaves = [(lower, 0, l, u)]
    serial = 1
    trace = [dict(splits=0, nodes=1, L=lower, U=upper, elapsed=time.monotonic() - begun)]
    reason = root["reason"]
    while True:
        if lower > upper:
            finite, reason = False, "inconsistent_bounds"
            break
        status = _status(lower, upper, finite, safety_margin)
        if status != "UNKNOWN":
            reason = status.lower()
            break
        if not finite:
            reason = root["reason"] if splits == 0 else "numerical_failure"
            break
        if root_only:
            reason = "root_only" if root["reason"] != "timeout" else "timeout"
            break
        if deadline is not None and time.monotonic() >= deadline:
            reason = "timeout"
            break
        if max_splits is not None and splits >= max_splits:
            reason = "max_splits"
            break
        import utils_BaB as UBAB

        parent_lower, _, parent_l, parent_u = leaves[0]
        axis = int(UBAB.get_best_axis(net, l.device, tc, ac, parent_l, parent_u,
                                     alpha, None, None, rule="widest"))
        if not bool(parent_u[0, axis] > parent_l[0, axis]):
            reason = "zero_width"
            break
        children_l, children_u = UBAB.subproblems(parent_l, parent_u, axis, int(n_branches))
        if any(torch.equal(children_l[j:j+1], parent_l) and torch.equal(children_u[j:j+1], parent_u)
               for j in range(n_branches)):
            reason = "unsplittable"
            break
        # Keep the parent until both children are bounded.
        pending = []
        interrupted = False
        for j in range(n_branches):
            if deadline is not None and time.monotonic() >= deadline:
                reason, interrupted = "timeout", True
                break
            child_l, child_u = children_l[j:j+1], children_u[j:j+1]
            child = optimize_box(net, tc, ac, child_l, child_u, alpha, checkpoints=(), **kwargs)
            improve_upper(child, child_l, child_u)
            attempted_nodes += 1
            if child["U"] < upper:
                upper, witness = child["U"], child["z"]
            if not child["finite"]:
                finite, reason, interrupted = False, "numerical_failure", True
                break
            if child["evaluations"] == 0:
                reason, interrupted = "timeout", True
                break
            pending.append((max(parent_lower, child["L"]), serial, child_l, child_u))
            serial += 1
        if interrupted:
            lower = leaves[0][0]
            trace.append(dict(splits=splits, nodes=nodes, L=lower, U=upper,
                              elapsed=time.monotonic() - begun, pending_children=len(pending)))
            break
        heapq.heappop(leaves)
        for entry in pending:
            heapq.heappush(leaves, entry)
        splits += 1
        nodes += n_branches
        lower = leaves[0][0]
        trace.append(dict(splits=splits, nodes=nodes, L=lower, U=upper, elapsed=time.monotonic() - begun))
    return dict(L=lower, U=upper, z=witness,
                status=_status(lower, upper, finite, safety_margin), reason=reason,
                finite=finite, root=root, root_raw=root["raw"], root_gap=root["gap"],
                raw=root["raw"], gap=root["gap"], steps=root["steps"],
                splits=splits, nodes=nodes, attempted_nodes=attempted_nodes,
                active_leaves=len(leaves), trace=trace, elapsed=time.monotonic() - begun,
                **attack_totals)
