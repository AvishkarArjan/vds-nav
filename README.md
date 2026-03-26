# VDS-NAV

Depth images guided open exploration for drones     
 implementation of the following paper: https://ieeexplore.ieee.org/document/11152316

***NOTE: This is just an unofficial implementation of the paper. No model was trained due to insufficient compute capacity.***

## Install

```bash
git clone https://github.com/AvishkarArjan/vds-nav.git
cd vds-nav
conda env create -f environment.yml
conda activate vds-nav
```

## Training

```bash
python train.py
```

## Visualize training

```tensorboard --logdir=results```
