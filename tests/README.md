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
python -m coverage run --source=autoopt,examples -m unittest
```

Generate the code coverage report using the below command.

```bash
python -m coverage report -m
```
A sample result is shown below

```bash
NName                                 Stmts   Miss  Cover   Missing
------------------------------------------------------------------
autoopt/__init__.py                      1      0   100%
autoopt/optim/__init__.py                6      0   100%
autoopt/optim/auto_adagrad.py           58     15    74%   46, 48, 61-64, 78, 85, 93-95, 98, 118-121
autoopt/optim/auto_adam.py              65     43    34%   53, 55, 57, 62-64, 78, 86, 91-145
autoopt/optim/auto_gauss_newton.py      69     47    32%   25, 27, 41-43, 57, 64, 69-134
autoopt/optim/auto_optimizer.py        103     33    68%   42-47, 70, 114, 118, 123-135, 147, 167, 178-185, 190, 193, 204-206
autoopt/optim/auto_sgd.py               46     24    48%   62, 72, 77-78, 83-110
autoopt/optim/gauss_newton.py           69     45    35%   25, 27, 29, 31, 38-40, 43, 57, 62, 67-124
autoopt/util/__init__.py                 0      0   100%
autoopt/util/error.py                    3      1    67%   25
autoopt/version.py                       1      1     0%   16
examples/__init__.py                     0      0   100%
examples/mnist.py                      144     67    53%   34-43, 56-69, 110, 113, 142, 145, 164, 171-193, 199-212, 218-225, 229
------------------------------------------------------------------
TOTAL                                  565    276    51%
```
