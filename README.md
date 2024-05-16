# Reinforcement Learning for Scheduling

This repo is meant as a starting point for exploring reinforcement learning techniques for solving scheduling problems.
Extends [Jumanji's JobShop environment](https://instadeepai.github.io/jumanji/environments/job_shop/) with dynamic features, support for benchmark instances and comparison against CP solvers.

## Installation

Setup a virtual environment with Python 3 and run:

```bash
pip install -r requirements.txt
```

MiniZinc is required for comparing to solutions of a CP solver.
For installation instructions see [https://www.minizinc.org/](https://www.minizinc.org).

## Usage

Configure the job shop environment and agent in [configs/env/job_shop.yaml](configs/env/job_shop.yaml).
Run the training:

```bash
python train.py
```
Alternatively, you can use the [hydra command line interface](https://hydra.cc/docs/intro/) for configuration. For example, the following command sets the learning rate of the agent:
```bash
python train.py env.a2c.learning_rate=1e-6
```

View metrics and schedules in tensorboard:
```bash
tensorboard --logdir .
```

## Standalone use of the constrained programming model

The CP model is written in the [MiniZinc language](https://www.minizinc.org/index.html).
It is located in [minizinc/job_shop.mzn](minizinc/job_shop.mzn).
To use a solver for an instance, modify and run:
```bash
python cp_solver.py
```

## Instance Data

The instance data in [instances](instances) comes from https://github.com/tamy0612/JSPLIB.