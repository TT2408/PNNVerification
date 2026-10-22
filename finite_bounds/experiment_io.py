import hashlib
import json
import math
import os
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
MODELS = json.loads((Path(__file__).with_name("models.json")).read_text())


def sha256(path):
    digest = hashlib.sha256()
    with open(path, "rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def save(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(f".{os.getpid()}.tmp")
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")
    temporary.replace(path)


def serializable(value):
    if isinstance(value, float) and not math.isfinite(value):
        return None
    if isinstance(value, dict):
        return {k: serializable(v) for k, v in value.items() if k != "z"}
    if isinstance(value, (list, tuple)):
        return [serializable(v) for v in value]
    return value


def prepare(output, identity, resume):
    manifest = output / "manifest.json"
    if manifest.exists():
        if not resume or json.loads(manifest.read_text()) != identity:
            raise ValueError(f"Existing run differs or --resume is missing: {output}")
    elif output.exists() and any(output.iterdir()):
        raise ValueError(f"Nonempty run directory without a manifest: {output}")
    else:
        save(manifest, identity)


def source_hashes():
    files = [ROOT / name for name in ("utils_NN.py", "utils_PN.py", "utils_BaB.py", "utils_zonotopes.py")]
    files += [Path(__file__).with_name(name) for name in
              ("experiment_io.py", "experiments.py", "solver.py", "curvature.py", "models.json")]
    return {str(path.relative_to(ROOT)): sha256(path) for path in files}


def load_model(path):
    import utils_NN as NN
    import utils_PN  # Full-object torch.save models refer to these classes.

    net = torch.load(path, map_location="cpu", weights_only=False)
    if not isinstance(net, torch.nn.Module):
        raise TypeError("Expected a trusted torch.save(model) file")
    net.eval().double()
    if isinstance(net, NN.NeuralNet) and getattr(net, "BN", False):
        probe = torch.linspace(0, 1, 3 * net.input_size, dtype=torch.float64).reshape(3, -1)
        with torch.no_grad():
            before = net(probe).clone()
            net = NN.fuse_BN_with_linear(net).eval().double()
            if not torch.allclose(before, net(probe), atol=1e-9, rtol=1e-9):
                raise RuntimeError("Batch-normalization fusion changed model outputs")
    for module in net.modules():
        if hasattr(module, "device"):
            module.device = torch.device("cpu")
        for name, value in list(vars(module).items()):
            if torch.is_tensor(value) and name not in module._parameters and name not in module._buffers:
                setattr(module, name, value.to(device="cpu", dtype=torch.float64))
    for parameter in net.parameters():
        parameter.requires_grad_(False)
    return net
