# Finite-computation experiments

`experiments.py` runs the three MNIST comparisons reported in the paper:

| Command | Comparison |
| --- | --- |
| `lower` | Finite PGD objective values and certified lower bounds, for three sigmoid networks and one quadratic PN |
| `alpha` | Numerical and certified alpha values for all 90 ordered class pairs of each model |
| `verify` | End-to-end VeriSmooth with both certificates, for the three sigmoid networks |

run
```sh
python -m finite_bounds.experiments lower --weights-root WEIGHTS --data-root DATA --output RESULTS
python -m finite_bounds.experiments alpha --weights-root WEIGHTS --output RESULTS
python -m finite_bounds.experiments verify --weights-root WEIGHTS --data-root DATA --output RESULTS
python -m finite_bounds.tables RESULTS
```