# FedDBO
FedDBO: Enhancing Federated Learning Aggregation with Dung Beetle Optimization

# Code will come soon.

Look forward to it.

# Datasets and Scenarios (Updating)

## Label Skew Scenario

- MNIST

IID + Unbalanced

```bash
python  ./data/generate_mnist.py iid - - # for iid and unbalanced scenario
```

IID + balanced

```bash
python ./data/generate_mnist.py iid balance - # for iid and balanced scenario
```

Non-IID + Unbalanced Pathological Scenario

```bash
python ./data/generate_mnist.py noniid - pat # for pathological noniid and unbalanced scenario
```

Non-IID + Unbalanced Practical Scenario

```bash
python ./data/generate_mnist.py noniid - dir # for practical noniid and unbalanced scenario
```

- CIFAR-100
  IID + Unbalanced

```bash
python  ./data/generate_cifar100.py iid - - # for iid and unbalanced scenario
```

IID + balanced

```bash
python ./data/generate_cifar100.py iid balance - # for iid and balanced scenario
```

Non-IID + Unbalanced Pathological Scenario

```bash
python ./data/generate_cifar100.py noniid - pat # for pathological noniid and unbalanced scenario
```

Non-IID + Unbalanced Practical Scenario

```bash
python ./data/generate_cifar100.py noniid - dir # for practical noniid and unbalanced scenario
```

- CIFAR-10
  IID + Unbalanced

```bash
python  ./data/generate_cifar10.py iid - - # for iid and unbalanced scenario
```

IID + balanced

```bash
python ./data/generate_cifar10.py iid balance - # for iid and balanced scenario
```

Non-IID + Unbalanced Pathological Scenario

```bash
python ./data/generate_cifar10.py noniid - pat # for pathological noniid and unbalanced scenario
```

Non-IID + Unbalanced Practical Scenario

```bash
python ./data/generate_cifar10.py noniid - dir # for practical noniid and unbalanced scenario
```

- F-MNIST

IID + Unbalanced

```bash
python  ./data/generate_fmnist.py iid - - # for iid and unbalanced scenario
```

IID + balanced

```bash
python ./data/generate_fmnist.py iid balance - # for iid and balanced scenario
```

Non-IID + Unbalanced Pathological Scenario

```bash
python ./data/generate_fmnist.py noniid - pat # for pathological noniid and unbalanced scenario
```

Non-IID + Unbalanced Practical Scenario

```bash
python ./data/generate_fmnist.py noniid - dir # for practical noniid and unbalanced scenario
```

- Digit5

IID + Unbalanced

```bash
python  ./data/generate_Digit5.py iid - - # for iid and unbalanced scenario
```

IID + balanced

```bash
python ./data/generate_Digit5.py iid balance - # for iid and balanced scenario
```

Non-IID + Unbalanced Pathological Scenario

```bash
python ./data/generate_Digit5.py noniid - pat # for pathological noniid and unbalanced scenario
```

Non-IID + Unbalanced Practical Scenario

```bash
python ./data/generate_Digit5.py noniid - dir # for practical noniid and unbalanced scenario
```

- Tiny-imagenet

IID + Unbalanced

```bash
python  ./data/generate_tiny_imagenet.py iid - - # for iid and unbalanced scenario
```

IID + balanced

```bash
python ./data/generate_tiny_imagenet.py iid balance - # for iid and balanced scenario
```

Non-IID + Unbalanced Pathological Scenario

```bash
python ./data/generate_tiny_imagenet.py noniid - pat # for pathological noniid and unbalanced scenario
```

Non-IID + Unbalanced Practical Scenario

```bash
python ./data/generate_tiny_imagenet.py noniid - dir # for practical noniid and unbalanced scenario
```

## Run

...