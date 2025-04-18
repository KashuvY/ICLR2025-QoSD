# DiffILO: Differentiable Integer Linear Programmin
This repository contains the implementation of DiffILO, an unsupervised learning approach for predicting solutions to Integer Linear Programs (ILPs).

Paper: https://openreview.net/pdf?id=FPfCUJTsCn

**Note:** This is the latest version, and we're still in the process of organizing and refining the code. Updates will follow. Feel free to reach out with any questions.

# Environment Setup
- Python environment
    - python 3.8
    - pytorch 2.3.0
    - torch-geometric 2.6
    - ecole 0.8.1
    - pyscipopt 4.4.0
    - gurobipy 10.0
    - tensorboardX

- MILP Solver
    - [Gurobi](https://www.gurobi.com/) 11.0.3. Academic License.

- Hydra
    - [Hydra](https://hydra.cc/docs/intro/) for managing hyperparameters and experiments.

To set up the environment, run the installation script:
```
bash scripts/environment.sh
```
Alternatively, create the environment from a file:
```
conda env create -f scripts/environment.yml
```

# Project Structure
The workspace is organized as follows:

```
DiffILO
├── conf/               # Hydra config files
├── data/               # Dataset directory (see below)
│   ├── CA/
│   │   ├── train/
│   │   └── test/
│   ├── IS/
│   │   ├── train/
│   │   └── test/
│   └── SC/
│       ├── train/
│       └── test/
├── scripts/            # Training/testing scripts
├── src/                # Core implementation
├── preprocess.py       # Data preprocessing
├── train.py            # Training entry point
├── test.py             # Evaluation entry point
└── README.md

```


# Usage
### 1. Preprocessing
To preprocess a dataset (e.g., `SC`), run:
```
python preprocess.py dataset=SC num_workers=50
```

### 2. Training DiffILO
To train the model with default settings:
```
python train.py dataset=SC cuda=0
```

### 3. Test DiffILO
To evaluate the trained model,
```
python test.py dataset=SC cuda=0
```

# Citation
If you find DiffILO useful or relevant to your research, please consider citing our paper.

```bibtex
@inproceedings{
    geng2025differentiable,
    title={Differentiable Integer Linear Programming},
    author={Zijie Geng and Jie Wang and Xijun Li and Fangzhou Zhu and Jianye HAO and Bin Li and Feng Wu},
    booktitle={The Thirteenth International Conference on Learning Representations},
    year={2025},
    url={https://openreview.net/forum?id=FPfCUJTsCn}
}
```