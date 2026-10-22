import argparse
import itertools
import json
import math
import platform
import signal
import time
from pathlib import Path

import torch
import numpy as np

from .curvature import CurvatureBounds
from .experiment_io import MODELS, load_model, prepare, save, serializable, sha256, source_hashes
from .solver import optimize_box, solve_property


def alpha_pair(net, curvature, target, rival):
    torch.manual_seed(0)
    def timeout(signum, frame):
        raise TimeoutError("Numerical alpha exceeded 30 seconds")
    previous = signal.signal(signal.SIGALRM, timeout)
    try:
        signal.alarm(30)
        if curvature.is_nn:
            numerical = float(net.singla_2020("cpu", target, rival)) / 2
        else:
            lower = torch.zeros(1, net.total_image_size)
            net.compute_min_eig(target, rival, lower, torch.ones_like(lower), "cpu")
            numerical = -float(net.min_eig)
    finally:
        signal.alarm(0)
        signal.signal(signal.SIGALRM, previous)
    signed = curvature.alpha(target, rival)
    certified = signed if curvature.is_nn else max(signed, curvature.alpha(rival, target))
    if not math.isfinite(numerical) or numerical <= 0 or not math.isfinite(certified):
        raise ValueError("Nonpositive or nonfinite alpha in relative comparison")
    return dict(target=target, rival=rival, numerical=numerical, certified=certified,
                verification_alpha=signed, relative_change=certified / numerical - 1)


def evaluate_image(net, curvature, dataset, image_id, case, args):
    started = time.monotonic()
    x, label = dataset[image_id]
    label = int(label)
    x = x.to(dtype=torch.float64).reshape(1, -1)
    with torch.no_grad():
        logits = net(x).flatten()
    if not torch.isfinite(logits).all():
        raise FloatingPointError("Nonfinite clean logits")
    predicted = int(logits.argmax())
    record = dict(image_id=image_id, label=label, predicted=predicted, properties=[])
    if predicted != label:
        return dict(record, status="CLEAN_INCORRECT")
    rivals = [int(r) for r in logits.argsort(descending=True) if int(r) != label]
    lower, upper = (x - case["eps"]).clamp(min=0), (x + case["eps"]).clamp(max=1)
    for rival in rivals:
        deadline = time.monotonic() + args.time_limit
        alpha = curvature.alpha(label, rival)
        options = dict(max_steps=args.steps, lr=1.0, gamma=0.95, deadline=deadline,
                       checkpoints=[args.steps], safety_margin=1e-10)
        if args.experiment == "lower":
            result = optimize_box(net, label, rival, lower, upper, alpha, **options)
            if not result["finite"] or result["steps"] != args.steps or result["evaluations"] != args.steps + 1:
                raise RuntimeError(f"Incomplete PGD trace: {result['reason']}")
        else:
            result = solve_property(net, label, rival, lower, upper, alpha,
                                    upper_steps=100, upper_lr=100.0, max_splits=100000, **options)
            if result["status"] == "ADVERSARIAL":
                witness = result["z"]
                with torch.no_grad():
                    output = net(witness)
                if not ((witness >= lower) & (witness <= upper)).all() or not output[0, label] - output[0, rival] < -1e-10:
                    raise RuntimeError("Invalid adversarial witness")
        result.update(rival=rival, alpha=alpha)
        record["properties"].append(serializable(result))
        if args.experiment == "verify" and result["status"] != "VERIFIED":
            break
    status = "DONE"
    if args.experiment == "verify":
        statuses = [p["status"] for p in record["properties"]]
        status = "VERIFIED" if len(statuses) == len(rivals) and all(s == "VERIFIED" for s in statuses) else statuses[-1]
    return dict(record, status=status, seconds=time.monotonic() - started)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("experiment", choices=["lower", "alpha", "verify"])
    parser.add_argument("--weights-root", type=Path, required=True)
    parser.add_argument("--data-root", type=Path, default=Path("data"))
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--models", nargs="+", choices=[m["id"] for m in MODELS])
    parser.add_argument("--count", type=int, default=1000)
    parser.add_argument("--steps", type=int, default=500)
    parser.add_argument("--time-limit", type=float, default=60.0)
    parser.add_argument("--shard-index", type=int, default=0)
    parser.add_argument("--num-shards", type=int, default=1)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    if not (0 < args.count <= 10000 and args.steps >= 0 and math.isfinite(args.time_limit)
            and args.time_limit > 0 and 0 <= args.shard_index < args.num_shards):
        parser.error("Invalid count, step budget, time limit, or shard")
    if args.experiment == "alpha" and args.num_shards != 1:
        parser.error("Alpha uses all 90 class pairs; do not shard this experiment")
    if args.experiment == "verify" and args.models and "quadratic" in args.models:
        parser.error("The end-to-end comparison uses the three sigmoid networks")
    torch.set_num_threads(1)
    torch.set_default_dtype(torch.float64)
    torch.manual_seed(0)
    np.random.seed(0)
    dataset = None
    if args.experiment != "alpha":
        from torchvision.datasets import MNIST
        from torchvision.transforms import ToTensor
        dataset = MNIST(str(args.data_root), train=False, download=True, transform=ToTensor())
    config = {k: str(v.resolve()) if isinstance(v, Path) else v for k, v in vars(args).items()
              if k not in ("models", "resume", "output", "weights_root")}
    environment = dict(python=platform.python_version(), platform=platform.platform(),
                       torch=torch.__version__, numpy=np.__version__, dtype="float64", threads=1)
    for case in MODELS:
        if args.models and case["id"] not in args.models:
            continue
        if args.experiment == "verify" and "reference" not in case:
            continue
        weights = args.weights_root / case["weights"]
        if not weights.is_file():
            weights = args.weights_root / Path(case["weights"]).name
        if sha256(weights) != case["sha256"]:
            raise ValueError(f"Weights differ from the experiment model: {weights}")
        output = args.output / args.experiment / case["id"] / f"shard_{args.shard_index:03d}"
        identity = dict(config=config, model=case, sources=source_hashes(), environment=environment)
        prepare(output, identity, args.resume)
        net = load_model(weights)
        curvature = CurvatureBounds(net)
        tasks = list(itertools.permutations(range(10), 2)) if args.experiment == "alpha" else range(args.shard_index, args.count, args.num_shards)
        for task in tasks:
            name = f"{task[0]}_{task[1]}" if args.experiment == "alpha" else f"{task:05d}"
            destination = output / "records" / (name + ".json")
            if destination.exists():
                if json.loads(destination.read_text())["status"] == "FAILED":
                    raise RuntimeError(f"Failure retained at {destination}; inspect it before restarting")
                continue
            try:
                result = (dict(alpha_pair(net, curvature, *task), status="DONE") if args.experiment == "alpha"
                          else evaluate_image(net, curvature, dataset, task, case, args))
            except Exception as exc:
                save(destination, dict(status="FAILED", error=f"{type(exc).__name__}: {exc}"))
                raise
            save(destination, result)
            print(f"{case['id']} {name}: {result['status']}", flush=True)


if __name__ == "__main__":
    main()
