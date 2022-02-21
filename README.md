# Graph WaveNet (Pytorch-lightning)

[Pytorch lightning](https://www.pytorchlightning.ai/) implementation of the original Graph WaveNet ([paper](https://arxiv.org/abs/1906.00121), [code](https://github.com/nnzhan/Graph-WaveNet)).

## 1. Dependencies

> **_NOTE:_** [Anaconda](https://docs.anaconda.com/anaconda/install/index.html) should be installed in the system.

### 1.1. Create a conda environment

```bash
conda env create -f environment.yml
```

### 1.2. Activate the environment

```bash
conda activate gwnet
```

## 2. Training

### 2.1. METR-LA dataset

```bash
python run.py --config=config/LA_gwnet.yaml --train
```

### 2.2. PEMS-BAY dataset

```bash
python run.py --config=config/BAY_gwnet.yaml --train
```

## 3. Test with pretrained weights

### 3.1. METR-LA

```bash
python run.py --config=config/LA_gwnet.yaml --test
```

#### 3.1.1. Test result (horizon 1 to 12)

```bash
Horizon 1 (5 min) - MAE: 2.23, RMSE: 3.83, MAPE: 5.36
Horizon 2 (10 min) - MAE: 2.51, RMSE: 4.61, MAPE: 6.25
Horizon 3 (15 min) - MAE: 2.70, RMSE: 5.13, MAPE: 6.93
Horizon 4 (20 min) - MAE: 2.85, RMSE: 5.54, MAPE: 7.51
Horizon 5 (25 min) - MAE: 2.98, RMSE: 5.89, MAPE: 7.97
Horizon 6 (30 min) - MAE: 3.09, RMSE: 6.17, MAPE: 8.37
Horizon 7 (35 min) - MAE: 3.18, RMSE: 6.41, MAPE: 8.73
Horizon 8 (40 min) - MAE: 3.26, RMSE: 6.63, MAPE: 9.05
Horizon 9 (45 min) - MAE: 3.34, RMSE: 6.81, MAPE: 9.35
Horizon 10 (50 min) - MAE: 3.40, RMSE: 6.97, MAPE: 9.60
Horizon 11 (55 min) - MAE: 3.46, RMSE: 7.12, MAPE: 9.81
Horizon 12 (60 min) - MAE: 3.52, RMSE: 7.25, MAPE: 10.04
Aggregation - MAE: 3.04, RMSE: 6.03, MAPE: 8.25
```

### 3.2. PEMS-BAY

```bash
python run.py --config=config/BAY_gwnet.yaml --test
```

#### 3.2.1. Test result (horizon 1 to 12)

```bash
Horizon 1 (5 min) - MAE: 0.86, RMSE: 1.54, MAPE: 1.65
Horizon 2 (10 min) - MAE: 1.13, RMSE: 2.22, MAPE: 2.26
Horizon 3 (15 min) - MAE: 1.31, RMSE: 2.76, MAPE: 2.74
Horizon 4 (20 min) - MAE: 1.46, RMSE: 3.18, MAPE: 3.14
Horizon 5 (25 min) - MAE: 1.57, RMSE: 3.51, MAPE: 3.46
Horizon 6 (30 min) - MAE: 1.66, RMSE: 3.78, MAPE: 3.73
Horizon 7 (35 min) - MAE: 1.73, RMSE: 3.99, MAPE: 3.95
Horizon 8 (40 min) - MAE: 1.80, RMSE: 4.17, MAPE: 4.13
Horizon 9 (45 min) - MAE: 1.85, RMSE: 4.30, MAPE: 4.29
Horizon 10 (50 min) - MAE: 1.90, RMSE: 4.42, MAPE: 4.43
Horizon 11 (55 min) - MAE: 1.94, RMSE: 4.52, MAPE: 4.56
Horizon 12 (60 min) - MAE: 1.99, RMSE: 4.63, MAPE: 4.69
Aggregation - MAE: 1.60, RMSE: 3.59, MAPE: 3.59
```

## 4. Tensorboard

It is possible to run tensorboard with saved logs.

### 4.1. METR-LA

```bash
tensorboard --logdir=experiments/la 
```

### 4.2. PEMS-BAY

```bash
tensorboard --logdir=experiments/bay
```
