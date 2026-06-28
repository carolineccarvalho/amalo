# %% ----------------------------------------------------------------------
# Preparacao dos dados: CIFAR-10 e CIFAR-100
#
# Mesmo espirito do bloco de dados do experimento FashionMNIST:
#   - normaliza com media/desvio do dataset
#   - usa um subconjunto para caber em tempo razoavel de treino (Colab/CPU)
#   - separa um "batch fixo" grande (hx, hy) so para estimar a Hessiana
#     sempre no mesmo ponto, igual ja era feito no script original
#
# Diferenca para o FashionMNIST: imagens RGB 32x32 (3 canais em vez de 1),
# por isso o modelo de exemplo aqui usa uma CNN pequena em vez de um MLP puro
# -- um MLP em 32x32x3 = 3072 entradas tambem funcionaria, mas a CNN deixa
# os resultados de sharpness mais comparaveis ao que se espera na literatura
# de visao (Foret et al. 2021 usa CNNs/ResNets, nao MLPs, para CIFAR).
# ------------------------------------------------------------------------
import torch
from torch.utils.data import DataLoader, Subset
import torchvision
import torchvision.transforms as T

device = "cuda" if torch.cuda.is_available() else "cpu"


# ============================================================
# CIFAR-10
# ============================================================
# Media/desvio padrao por canal (valores usuais na literatura para CIFAR-10)
CIFAR10_MEAN = (0.4914, 0.4822, 0.4465)
CIFAR10_STD  = (0.2470, 0.2435, 0.2616)


def get_cifar10_loaders(n_train=16000, n_test=4000, n_hess=1000,
                         batch_size_train=128, batch_size_test=512,
                         data_root="./data", seed=0):
    """
    Prepara DataLoaders de treino/teste para CIFAR-10, alem de um batch fixo
    (hx, hy) para estimar lambda_max sempre no mesmo ponto.

    Por padrao usa subconjuntos (n_train/n_test) para rodar rapido fora de
    GPU dedicada -- aumente esses valores se tiver tempo/hardware.

    Retorna: train_loader, test_loader, hx, hy, n_classes, input_shape
    """
    torch.manual_seed(seed)

    tf_train = T.Compose([T.ToTensor(), T.Normalize(CIFAR10_MEAN, CIFAR10_STD)])
    tf_test  = T.Compose([T.ToTensor(), T.Normalize(CIFAR10_MEAN, CIFAR10_STD)])

    train_full = torchvision.datasets.CIFAR10(root=data_root, train=True,  download=True, transform=tf_train)
    test_full  = torchvision.datasets.CIFAR10(root=data_root, train=False, download=True, transform=tf_test)

    n_train = min(n_train, len(train_full))
    n_test  = min(n_test,  len(test_full))
    train_ds = Subset(train_full, range(n_train))
    test_ds  = Subset(test_full,  range(n_test))

    train_loader = DataLoader(train_ds, batch_size=batch_size_train, shuffle=True)
    test_loader  = DataLoader(test_ds,  batch_size=batch_size_test,  shuffle=False)

    # batch fixo para estimar a Hessiana sempre no mesmo ponto (igual ao FashionMNIST)
    n_hess = min(n_hess, n_train)
    hx = torch.stack([train_ds[i][0] for i in range(n_hess)]).to(device)
    hy = torch.tensor([train_ds[i][1] for i in range(n_hess)]).to(device)

    return train_loader, test_loader, hx, hy, 10, (3, 32, 32)


# ============================================================
# CIFAR-100
# ============================================================
# Media/desvio padrao por canal (valores usuais na literatura para CIFAR-100)
CIFAR100_MEAN = (0.5071, 0.4865, 0.4409)
CIFAR100_STD  = (0.2673, 0.2564, 0.2762)


def get_cifar100_loaders(n_train=16000, n_test=4000, n_hess=1000,
                          batch_size_train=128, batch_size_test=512,
                          data_root="./data", seed=0):
    """
    Mesmo papel de get_cifar10_loaders, mas para CIFAR-100 (100 classes).
    """
    torch.manual_seed(seed)

    tf_train = T.Compose([T.ToTensor(), T.Normalize(CIFAR100_MEAN, CIFAR100_STD)])
    tf_test  = T.Compose([T.ToTensor(), T.Normalize(CIFAR100_MEAN, CIFAR100_STD)])

    train_full = torchvision.datasets.CIFAR100(root=data_root, train=True,  download=True, transform=tf_train)
    test_full  = torchvision.datasets.CIFAR100(root=data_root, train=False, download=True, transform=tf_test)

    n_train = min(n_train, len(train_full))
    n_test  = min(n_test,  len(test_full))
    train_ds = Subset(train_full, range(n_train))
    test_ds  = Subset(test_full,  range(n_test))

    train_loader = DataLoader(train_ds, batch_size=batch_size_train, shuffle=True)
    test_loader  = DataLoader(test_ds,  batch_size=batch_size_test,  shuffle=False)

    n_hess = min(n_hess, n_train)
    hx = torch.stack([train_ds[i][0] for i in range(n_hess)]).to(device)
    hy = torch.tensor([train_ds[i][1] for i in range(n_hess)]).to(device)

    return train_loader, test_loader, hx, hy, 100, (3, 32, 32)


if __name__ == "__main__":

    print("device:", device)

    tl, te, hx, hy, n_classes, shape = get_cifar10_loaders(n_train=2000, n_test=500, n_hess=200)
    print(f"CIFAR-10  -> n_classes={n_classes} input_shape={shape} "
          f"train_batches={len(tl)} test_batches={len(te)} hx={hx.shape} hy={hy.shape}")

    tl, te, hx, hy, n_classes, shape = get_cifar100_loaders(n_train=2000, n_test=500, n_hess=200)
    print(f"CIFAR-100 -> n_classes={n_classes} input_shape={shape} "
          f"train_batches={len(tl)} test_batches={len(te)} hx={hx.shape} hy={hy.shape}")
