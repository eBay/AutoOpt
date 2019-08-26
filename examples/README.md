# AutoOpt Examples

This directory contains sample code that utilizes the AutoOpt package 
for ML training.

 - `mnist.py`: PyTorch based MNIST model training that lets compare between different
 optimizers.

For the regular SGD optimizer just run the script without any parameters: 

```bash
$ python mnist.py 
Train Epoch: 1 [0/60000 (0%)]	Loss: 2.296416
Train Epoch: 1 [6400/60000 (11%)]	Loss: 1.841516
Train Epoch: 1 [12800/60000 (21%)]	Loss: 1.103357
Train Epoch: 1 [19200/60000 (32%)]	Loss: 0.648390
Train Epoch: 1 [25600/60000 (43%)]	Loss: 0.492310
Train Epoch: 1 [32000/60000 (53%)]	Loss: 0.378769
Train Epoch: 1 [38400/60000 (64%)]	Loss: 0.489791
Train Epoch: 1 [44800/60000 (75%)]	Loss: 0.381028
Train Epoch: 1 [51200/60000 (85%)]	Loss: 0.328171
Train Epoch: 1 [57600/60000 (96%)]	Loss: 0.332823

Test set: Average loss: 0.0003, Accuracy: 9040/10000 (90%)
```

You can specify the optimizer using the `--optimizer` parameter. For example, to use
the AutoSGD optimizer, specify `auto-sgd` in the command line:

```bash
$ python mnist.py --optimizer auto-sgd
Train Epoch: 1 [0/60000 (0%)]	Loss: 2.317884
Train Epoch: 1 [6400/60000 (11%)]	Loss: 0.364227
Train Epoch: 1 [12800/60000 (21%)]	Loss: 0.432900
Train Epoch: 1 [19200/60000 (32%)]	Loss: 0.140982
Train Epoch: 1 [25600/60000 (43%)]	Loss: 0.099320
Train Epoch: 1 [32000/60000 (53%)]	Loss: 0.142141
Train Epoch: 1 [38400/60000 (64%)]	Loss: 0.116720
Train Epoch: 1 [44800/60000 (75%)]	Loss: 0.224322
Train Epoch: 1 [51200/60000 (85%)]	Loss: 0.125071
Train Epoch: 1 [57600/60000 (96%)]	Loss: 0.086979

Test set: Average loss: 0.0001, Accuracy: 9580/10000 (95%)
```

Type `--help` for all available command line parameters.
