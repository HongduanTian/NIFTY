# NIFTY
This is a rough reproduction of [''Towards a Unified Framework for Fair and Stable Graph Representation Learning''](https://arxiv.org/pdf/2102.13186.pdf) on German dataset.

The original code provided by the author is [here](https://github.com/chirag126/nifty)

## Experiments
1. For vanilla GCN, run `python main.py --model gcn`:<br> ![pic1](./gcn_res.JPG)
2. For NIFTY-GCN, run `python main.py --model niftygcn --sim --lipschitz`: ![pic2](./nifty_res.JPG)
