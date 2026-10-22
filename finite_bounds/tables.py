"""Summarize complete experiment runs into the three comparison tables."""
import argparse
import csv
import itertools
import json
import math
from collections import defaultdict
from pathlib import Path

import numpy as np


def collect(root):
    groups = defaultdict(list)
    for path in sorted(root.glob("*/*/shard_*/manifest.json")):
        manifest = json.loads(path.read_text())
        groups[(manifest["config"]["experiment"], manifest["model"]["id"])].append((path.parent, manifest))
    if not groups:
        raise ValueError("No experiment manifests found")
    for (experiment, model), shards in sorted(groups.items()):
        identity, records = None, {}
        for directory, manifest in shards:
            config = manifest["config"]
            common = dict(manifest, config={k: v for k, v in config.items() if k not in ("shard_index", "num_shards")})
            if identity is not None and identity != common:
                raise ValueError(f"Incompatible shards for {experiment}/{model}")
            identity = common
            for path in sorted((directory / "records").glob("*.json")):
                row = json.loads(path.read_text())
                if row["status"] == "FAILED":
                    raise ValueError(f"Failed record: {path}")
                key = (row["target"], row["rival"]) if experiment == "alpha" else row["image_id"]
                if key in records:
                    raise ValueError(f"Duplicate record: {path}")
                if experiment != "alpha" and key % config["num_shards"] != config["shard_index"]:
                    raise ValueError(f"Image belongs to a different shard: {path}")
                records[key] = row
        config, case = identity["config"], identity["model"]
        expected = set(itertools.permutations(range(10), 2)) if experiment == "alpha" else set(range(config["count"]))
        if set(records) != expected:
            raise ValueError(f"Incomplete or unexpected records for {experiment}/{model}: {len(records)}/{len(expected)}")
        yield experiment, case, config, list(records.values())


def summarize(experiment, case, config, records):
    row = dict(model=case["label"])
    if experiment == "alpha":
        if any(p["status"] != "DONE" or not all(math.isfinite(p[k]) for k in
               ("numerical", "certified", "relative_change")) for p in records):
            raise ValueError("Incomplete alpha comparison")
        return dict(row, pairs=len(records), numerical_median=float(np.median([p["numerical"] for p in records])),
                    certified_median=float(np.median([p["certified"] for p in records])),
                    max_abs_relative_change=max(abs(p["relative_change"]) for p in records))
    n = len(records)
    clean = [p for p in records if p["predicted"] == p["label"]]
    row.update(n=n, steps=config["steps"], clean_correct=len(clean))
    if experiment == "lower":
        gaps, numerical, certified = [], 0, 0
        for image in clean:
            properties = image["properties"]
            if ({p["rival"] for p in properties} != set(range(10)) - {image["label"]} or len(properties) != 9
                    or any(not p["finite"] or p["steps"] != config["steps"] or p["evaluations"] != config["steps"] + 1
                           or not all(isinstance(p[k], (int, float)) and math.isfinite(p[k]) for k in ("raw", "L", "gap")) for p in properties)):
                raise ValueError(f"Incomplete lower-bound trace for image {image['image_id']}")
            numerical += all(p["raw"] > 0 for p in properties)
            certified += all(p["L"] > 0 for p in properties)
            gaps.extend(p["gap"] for p in properties)
        row.update(numerical_positive_percent=100 * numerical / n, certified_positive_percent=100 * certified / n,
                   gap_p95=float(np.percentile(gaps, 95)) if gaps else None)
    else:
        if any(p["status"] not in ("VERIFIED", "ADVERSARIAL", "UNKNOWN", "CLEAN_INCORRECT") for p in records):
            raise ValueError("Unexpected verification status")
        row.update(time_limit=config["time_limit"], verified=sum(p["status"] == "VERIFIED" for p in records))
        row["VeriSmooth_certified"] = 100 * row["verified"] / n
        if n == 1000 and config["steps"] == 500 and config["time_limit"] == 60:
            row.update(case["reference"])
    return row


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("results", type=Path)
    args = parser.parse_args()
    tables = defaultdict(list)
    for experiment, case, config, records in collect(args.results):
        tables[experiment].append(summarize(experiment, case, config, records))
    for experiment, rows in tables.items():
        path = args.results / f"{experiment}.csv"
        with path.open("w", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=list(dict.fromkeys(k for row in rows for k in row)))
            writer.writeheader()
            writer.writerows(rows)
        print(path)


if __name__ == "__main__":
    main()
