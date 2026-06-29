"""Carregamento e preprocessamento dos datasets.

Cada `load_*` devolve um `DataBundle` padronizado. Decisões de preparação:
- imputação (mediana/moda) e StandardScaler são ajustados SÓ no treino e depois
  aplicados a val/teste, para não vazar informação do conjunto de avaliação;
- para imagens, mantemos um subconjunto fixo `hess_inputs/hess_targets` (1000
  amostras) — calcular λ_max sobre todo o treino seria caro e desnecessário;
- `random_state=SEED` em todos os splits garante reprodutibilidade.
"""

from dataclasses import dataclass

import pandas as pd
import torch
import torchvision
import torchvision.transforms as T
from sklearn.datasets import load_iris as _sk_load_iris
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader, Subset, TensorDataset

SEED = 42

# Médias/desvios por canal de cada dataset de imagem (normalização padrão).
FASHIONMNIST_MEAN, FASHIONMNIST_STD = (0.286,), (0.353,)
CIFAR10_MEAN, CIFAR10_STD = (0.4914, 0.4822, 0.4465), (0.2470, 0.2435, 0.2616)
CIFAR100_MEAN, CIFAR100_STD = (0.5071, 0.4865, 0.4409), (0.2673, 0.2564, 0.2762)

ROOTPATH='/content/drive/MyDrive/amalo'

@dataclass
class DataBundle:
    train_loader: DataLoader
    val_loader: DataLoader
    test_loader: DataLoader
    hess_inputs: torch.Tensor       # tensores para λ_max / geometria (treino ou subconjunto)
    hess_targets: torch.Tensor
    input_dim: int = None           # entrada achatada (MLP); None para CNN
    num_classes: int = None         # 1 (binário) / 3 / 10 / 100
    input_shape: tuple = None       # (C,H,W) para imagens


def _extract_tensors(dataset, n):
    """Empilha as n primeiras amostras de um dataset torchvision em tensores."""
    X = torch.stack([dataset[i][0] for i in range(n)])
    y = torch.tensor([dataset[i][1] for i in range(n)], dtype=torch.long)
    return X, y


def load_titanic(batch_size=32, data_path=ROOTPATH+'/data/Titanic/Titanic-Dataset.csv'):
    df = pd.read_csv(data_path)
    df = df.drop(columns=['PassengerId', 'Name', 'Ticket', 'Cabin']).copy()

    X = df.drop(columns=['Survived'])
    y = df['Survived'].astype(float)

    X_train_raw, X_temp_raw, y_train, y_temp = train_test_split(
        X, y, test_size=0.4, stratify=y, random_state=SEED)
    X_val_raw, X_test_raw, y_val, y_test = train_test_split(
        X_temp_raw, y_temp, test_size=0.5, stratify=y_temp, random_state=SEED)

    # Estatísticas de imputação calculadas só no treino.
    categorical_cols = ['Sex', 'Embarked']
    age_median = X_train_raw['Age'].median()
    fare_median = X_train_raw['Fare'].median()
    embarked_mode = X_train_raw['Embarked'].mode()[0]

    def prep(split):
        split = split.copy()
        split['Age'] = split['Age'].fillna(age_median)
        split['Fare'] = split['Fare'].fillna(fare_median)
        split['Embarked'] = split['Embarked'].fillna(embarked_mode)
        split = pd.get_dummies(split, columns=categorical_cols, drop_first=True)
        return split.astype(float)

    X_train_pre = prep(X_train_raw)
    # reindex alinha as colunas dummy de val/teste às do treino (categorias ausentes -> 0).
    X_val_pre = prep(X_val_raw).reindex(columns=X_train_pre.columns, fill_value=0.0)
    X_test_pre = prep(X_test_raw).reindex(columns=X_train_pre.columns, fill_value=0.0)

    scaler = StandardScaler()
    X_train = torch.tensor(scaler.fit_transform(X_train_pre), dtype=torch.float32)
    X_val = torch.tensor(scaler.transform(X_val_pre), dtype=torch.float32)
    X_test = torch.tensor(scaler.transform(X_test_pre), dtype=torch.float32)
    y_train_t = torch.tensor(y_train.values, dtype=torch.float32).unsqueeze(1)
    y_val_t = torch.tensor(y_val.values, dtype=torch.float32).unsqueeze(1)
    y_test_t = torch.tensor(y_test.values, dtype=torch.float32).unsqueeze(1)

    train_loader = DataLoader(TensorDataset(X_train, y_train_t), batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(TensorDataset(X_val, y_val_t), batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(TensorDataset(X_test, y_test_t), batch_size=batch_size, shuffle=False)

    input_dim = X_train.shape[1]
    print(f'Titanic: input_dim={input_dim}  train={len(y_train)}  val={len(y_val)}  test={len(y_test)}')
    return DataBundle(train_loader, val_loader, test_loader,
                      hess_inputs=X_train, hess_targets=y_train_t,
                      input_dim=input_dim, num_classes=1, input_shape=None)


def load_iris(batch_size=16):
    iris = _sk_load_iris()
    X_raw = pd.DataFrame(iris.data, columns=iris.feature_names)
    y = iris.target

    X_tr_raw, X_tmp, y_tr, y_tmp = train_test_split(
        X_raw, y, test_size=0.4, stratify=y, random_state=SEED)
    X_val_raw, X_te_raw, y_val, y_te = train_test_split(
        X_tmp, y_tmp, test_size=0.5, stratify=y_tmp, random_state=SEED)

    scaler = StandardScaler()
    X_tr = torch.tensor(scaler.fit_transform(X_tr_raw), dtype=torch.float32)
    X_val = torch.tensor(scaler.transform(X_val_raw), dtype=torch.float32)
    X_te = torch.tensor(scaler.transform(X_te_raw), dtype=torch.float32)
    y_tr_t = torch.tensor(y_tr, dtype=torch.long)
    y_val_t = torch.tensor(y_val, dtype=torch.long)
    y_te_t = torch.tensor(y_te, dtype=torch.long)

    train_loader = DataLoader(TensorDataset(X_tr, y_tr_t), batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(TensorDataset(X_val, y_val_t), batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(TensorDataset(X_te, y_te_t), batch_size=batch_size, shuffle=False)

    input_dim = X_tr.shape[1]
    print(f'Iris: input_dim={input_dim}  train={len(y_tr)}  val={len(y_val)}  test={len(y_te)}')
    return DataBundle(train_loader, val_loader, test_loader,
                      hess_inputs=X_tr, hess_targets=y_tr_t,
                      input_dim=input_dim, num_classes=3, input_shape=None)


def _load_image(name, ds_cls, mean, std, num_classes, *, n_train, n_test,
                batch_train, batch_eval, val_fraction, n_hess, for_mlp, data_root):
    tf = T.Compose([T.ToTensor(), T.Normalize(mean, std)])
    train_full = ds_cls(root=data_root, train=True, download=True, transform=tf)
    test_full = ds_cls(root=data_root, train=False, download=True, transform=tf)

    train_ds = Subset(train_full, range(n_train))
    test_ds = Subset(test_full, range(n_test))
    X_all, y_all = _extract_tensors(train_ds, n_train)
    X_te, y_te = _extract_tensors(test_ds, n_test)

    X_tr, X_val, y_tr, y_val = train_test_split(
        X_all, y_all, test_size=val_fraction, random_state=SEED)

    train_loader = DataLoader(TensorDataset(X_tr, y_tr), batch_size=batch_train, shuffle=True)
    val_loader = DataLoader(TensorDataset(X_val, y_val), batch_size=batch_eval, shuffle=False)
    test_loader = DataLoader(TensorDataset(X_te, y_te), batch_size=batch_eval, shuffle=False)

    input_dim = X_tr.shape[1] * X_tr.shape[2] * X_tr.shape[3]  # achatado, p/ MLP
    input_shape = tuple(X_tr.shape[1:])
    print(f'{name}: shape={input_shape}  train={len(y_tr)}  val={len(y_val)}  test={len(y_te)}')
    return DataBundle(train_loader, val_loader, test_loader,
                      hess_inputs=X_tr[:n_hess], hess_targets=y_tr[:n_hess],
                      input_dim=input_dim if for_mlp else None,
                      num_classes=num_classes, input_shape=input_shape)


def load_fashion_mnist(n_train=6000, n_test=1500, batch_train=128, batch_eval=256,
                       val_fraction=0.15, n_hess=1000, data_root=ROOTPATH+'/data'):
    return _load_image('FashionMNIST', torchvision.datasets.FashionMNIST,
                       FASHIONMNIST_MEAN, FASHIONMNIST_STD, num_classes=10,
                       n_train=n_train, n_test=n_test, batch_train=batch_train,
                       batch_eval=batch_eval, val_fraction=val_fraction, n_hess=n_hess,
                       for_mlp=True, data_root=data_root)


def load_cifar10(n_train=6000, n_test=1500, batch_train=256, batch_eval=512,
                 val_fraction=0.15, n_hess=1000, data_root=ROOTPATH+'/data'):
    return _load_image('CIFAR-10', torchvision.datasets.CIFAR10,
                       CIFAR10_MEAN, CIFAR10_STD, num_classes=10,
                       n_train=n_train, n_test=n_test, batch_train=batch_train,
                       batch_eval=batch_eval, val_fraction=val_fraction, n_hess=n_hess,
                       for_mlp=False, data_root=data_root)


def load_cifar100(n_train=6000, n_test=1500, batch_train=256, batch_eval=512,
                  val_fraction=0.15, n_hess=1000, data_root=ROOTPATH+'/data'):
    return _load_image('CIFAR-100', torchvision.datasets.CIFAR100,
                       CIFAR100_MEAN, CIFAR100_STD, num_classes=100,
                       n_train=n_train, n_test=n_test, batch_train=batch_train,
                       batch_eval=batch_eval, val_fraction=val_fraction, n_hess=n_hess,
                       for_mlp=False, data_root=data_root)
