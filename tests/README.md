# Tests

This directory contains the unit tests.

### Unit Tests

Please change directory to the root dir of this repo and invoke the unit tests as follows:

```bash
python -m unittest
```

### Code Coverage

For code coverage, `coverage` package should be installed.

```bash
pip3 install coverage
```

Run the below command to run the unit tests and to gather coverage data for the autoopt module.

```bash
python -m coverage run --source=autoopt -m unittest
```

Generate the code coverage report using the below command.

```bash
python -m coverage report -m
```
A sample result is shown below

```bash
Name                                 Stmts   Miss  Cover   Missing
------------------------------------------------------------------
autoopt/__init__.py                      1      0   100%
autoopt/optim/__init__.py                6      0   100%
autoopt/optim/auto_adagrad.py           47     41    13%   17-30, 33-36, 46-80
autoopt/optim/auto_adam.py              65     58    11%   23-30, 33-35, 45-117
autoopt/optim/auto_gauss_newton.py      66     59    11%   9-23, 26-28, 38-116
autoopt/optim/auto_optimizer.py        100     60    40%   27-32, 55, 73, 86-88, 97-115, 126-130, 145-188
autoopt/optim/auto_sgd.py               46     39    15%   44-97
autoopt/optim/gauss_newton.py           69     61    12%   9-20, 23-25, 28, 38-111
autoopt/util/__init__.py                 0      0   100%
autoopt/util/error.py                    3      1    67%   10
autoopt/version.py                       1      1     0%   1
------------------------------------------------------------------
TOTAL                                  404    320    21%
```