# M3NSCT5
The code base for our paper （accepted by the Journal of Systems and Software）
"**[Diverse Title Generation for Stack Overflow Posts with Multiple-Sampling-Enhanced Transformer](https://doi.org/10.1016/j.jss.2023.111672)**"

M3NSCT5 is a hybrid method combining the Maximal Marginal Multiple Nucleus Sampling strategy and the CodeT5 model, which is proposed to tackle the ambiguity issue when generating Stack Overflow post titles from code snippets.

![framework](./figs/framework.png)

### Update🐇
- (Feb 26, 2023) Due to bandwidth limitation, the datasets included in this repo are currently not accessible. Therefore, we publish the data to [Zenodo](https://zenodo.org/record/7022467#.Y_txzptByHs).

### Environment
Before you run the code, make sure you have prepared the necessary environment:
```
conda create -n m3nsct5 python=3.7
conda activate m3nsct5
pip install -r requirements.txt
```

### Dataset and Models

Here is the directory structure of our repository:

```
├─main.py  # entry_point
├─mytrainer.py  # top-p + CodeT5
├─dataset  # the train/val/test data
├─metrics
│   ├─myrouge.py  # wrapper of rouge metric
│   ├─ranking.py  # maximal marginal ranking
│   └─scorer.py  # entry for evaluation
│
└─scripts
    ├─codet5_sample.sbatch  # example reference script
    └─codet5_train.sbatch  # example train script
```

#### Citation
If you find this work inspiring for your research, please cite our paper:
```
@article{ZHANG2023111672,
title = {Diverse title generation for Stack Overflow posts with multiple-sampling-enhanced transformer},
journal = {Journal of Systems and Software},
volume = {200},
pages = {111672},
year = {2023},
issn = {0164-1212},
doi = {https://doi.org/10.1016/j.jss.2023.111672},
url = {https://www.sciencedirect.com/science/article/pii/S0164121223000675},
author = {Fengji Zhang and Jin Liu and Yao Wan and Xiao Yu and Xiao Liu and Jacky Keung},
}
```
