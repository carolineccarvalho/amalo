# %% ----------------------------------------------------------------------
# Experimento: sharpness (lambda_max) vs generalizacao sob SAM, variando rho
# CIFAR-10 / CIFAR-100 / ImageNet + CNN pequena.
# HVP por Pearlmutter (reverse-over-reverse) + metodo da potencia.
#
# Mesma estrutura/principios do script original de FashionMNIST
# (loss_geometria / fashionmnist sharpness experiment):
#   - SAM (Foret et al. 2021) com SGD como otimizador base
#   - lambda_max(H) calculado via HVP (Pearlmutter) + power iteration,
#     sempre no mesmo batch fixo (hx, hy), para comparar os rhos de forma justa
#   - varios valores de rho, incluindo rho=0 (SGD puro, sem SAM)
#   - 3 graficos no final: lambda_max x epoca, gap x lambda_max, e
#     lambda_max(rho)/gap(rho) em eixos gemeos
#
# O que muda em relacao ao FashionMNIST:
#   - dataset escolhido via DATASET (cifar10 / cifar100 / imagenet)
#   - modelo: MLP simples nao faz sentido em imagens RGB maiores, entao
#     trocamos por uma CNN pequena (mesma ideia de "modelo simples o
#     suficiente para rodar rapido", so que com convolucoes)
#   - imports de cifar_datasets.py / imagenet_dataset.py em vez de
#     baixar FashionMNIST diretamente
#
# Como o calculo de lambda_max precisa de create_graph=True (grafo de
# segunda ordem), o custo de memoria/tempo cresce bastante com o tamanho do
# modelo e da imagem. Por isso os defaults aqui sao conservadores (poucas
# epocas, subconjunto pequeno do dataset, ImageNet em 64x64 em vez de
# 224x224) -- ajuste IMG_SIZE/N_TRAIN/EPOCHS de acordo com o hardware
# disponivel.
# ------------------------------------------------------------------------
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt

from cifar_datasets import get_cifar10_loaders, get_cifar100_loaders
from imagenet_dataset import get_imagenet_loaders, get_tiny_imagenet_loaders

device = "cuda" if torch.cuda.is_available() else "cpu"
torch.manual_seed(0)
print("device:", device)

# ---- escolha do dataset -------------------------------------------------
# "cifar10", "cifar100", "imagenet" (precisa de root local) ou
# "tiny_imagenet" (download automatico, util para testar o pipeline)
DATASET = "cifar10"
IMAGENET_ROOT = "./data/imagenet"   # so usado se DATASET == "imagenet"

N_TRAIN, N_TEST, N_HESS = 8000, 2000, 500   # subconjunto p/ rodar rapido
BATCH_TRAIN, BATCH_TEST = 128, 512

if DATASET == "cifar10":
    train_loader, test_loader, hx, hy, n_classes, input_shape = get_cifar10_loaders(
        n_train=N_TRAIN, n_test=N_TEST, n_hess=N_HESS,
        batch_size_train=BATCH_TRAIN, batch_size_test=BATCH_TEST,
    )
elif DATASET == "cifar100":
    train_loader, test_loader, hx, hy, n_classes, input_shape = get_cifar100_loaders(
        n_train=N_TRAIN, n_test=N_TEST, n_hess=N_HESS,
        batch_size_train=BATCH_TRAIN, batch_size_test=BATCH_TEST,
    )
elif DATASET == "imagenet":
    train_loader, test_loader, hx, hy, n_classes, input_shape = get_imagenet_loaders(
        root=IMAGENET_ROOT, n_train=N_TRAIN, n_test=N_TEST, n_hess=N_HESS,
        batch_size_train=64, batch_size_test=256, img_size=64,  # 64x64 por custo computacional
    )
elif DATASET == "tiny_imagenet":
    train_loader, test_loader, hx, hy, n_classes, input_shape = get_tiny_imagenet_loaders(
        n_train=N_TRAIN, n_test=N_TEST, n_hess=N_HESS,
        batch_size_train=64, batch_size_test=256,
    )
else:
    raise ValueError(f"DATASET desconhecido: {DATASET}")

print(f"dataset={DATASET}  n_classes={n_classes}  input_shape={input_shape}  "
      f"train_batches={len(train_loader)}  test_batches={len(test_loader)}")


def make_model(input_shape, n_classes):
    c, h, w = input_shape

    def conv_block(in_ch, out_ch):
        return nn.Sequential(
            nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
        )

    model = nn.Sequential(
        conv_block(c, 32),     # h/2  x w/2
        conv_block(32, 64),    # h/4  x w/4
        conv_block(64, 128),   # h/8  x w/8
        nn.AdaptiveAvgPool2d(1),
        nn.Flatten(),
        nn.Linear(128, n_classes),
    ).to(device)
    return model


_tmp_model = make_model(input_shape, n_classes)
n_params = sum(p.numel() for p in _tmp_model.parameters())
print(f"n_params = {n_params}  (Hessiana densa teria {n_params**2:.2e} entradas)")
del _tmp_model


# ---- SAM ---------------------------------------------------------------
# Identico ao SAM do script original (Foret et al. 2021, base SGD).
class SAM(torch.optim.Optimizer):
    """Sharpness-Aware Minimization (Foret et al. 2021), base SGD."""
    def __init__(self, params, base_optimizer, rho=0.05, **kw):
        defaults = dict(rho=rho, **kw)
        super().__init__(params, defaults)
        self.base = base_optimizer(self.param_groups, **kw)
        self.param_groups = self.base.param_groups

    @torch.no_grad()
    def first_step(self):
        grad_norm = torch.norm(torch.stack([
            p.grad.norm(2) for g in self.param_groups for p in g["params"] if p.grad is not None]), 2)
        for g in self.param_groups:
            scale = g["rho"] / (grad_norm + 1e-12)
            for p in g["params"]:
                if p.grad is None: continue
                e_w = p.grad * scale
                p.add_(e_w); self.state[p]["e_w"] = e_w   # sobe para o pico

    @torch.no_grad()
    def second_step(self):
        for g in self.param_groups:
            for p in g["params"]:
                if p.grad is None: continue
                p.sub_(self.state[p]["e_w"])               # volta ao ponto original
        self.base.step()


# ---- HVP (Pearlmutter) + metodo da potencia ----------------------------
# Identico ao power_iteration do script original.
def power_iteration(model, x, y, n_iter=20):
    """lambda_max(H) usando apenas HVPs. H = Hessiana da perda em (x,y)."""
    params = [p for p in model.parameters() if p.requires_grad]
    loss = F.cross_entropy(model(x), y)
    grads = torch.autograd.grad(loss, params, create_graph=True)   # mantem o grafo
    v = [torch.randn_like(p) for p in params]
    nrm = math.sqrt(sum((vi**2).sum().item() for vi in v))
    v = [vi / nrm for vi in v]
    lam = 0.0
    for _ in range(n_iter):
        dot = sum((g*vi).sum() for g, vi in zip(grads, v))         # <grad, v>
        Hv = torch.autograd.grad(dot, params, retain_graph=True)   # grad de <grad,v> = H v
        lam = sum((vi*hv).sum().item() for vi, hv in zip(v, Hv))   # quociente de Rayleigh
        nrm = math.sqrt(sum((hv**2).sum().item() for hv in Hv))
        if nrm < 1e-12: break
        v = [hv / nrm for hv in Hv]
    return lam


# ---- avaliacao -----------------------------------------------------------
@torch.no_grad()
def accuracy(model, loader):
    model.eval(); correct = total = 0
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        correct += (model(x).argmax(1) == y).sum().item(); total += y.numel()
    return correct / total


# %% ----------------------------------------------------------------------
# Treino: para cada rho, treina e registra lambda_max por epoca
#
# Igual ao FashionMNIST: lambda_max e calculado todo epoch, sempre sobre o
# mesmo batch fixo (hx, hy), permitindo plotar a curva lambda_max x epoca
# e comparar entre diferentes valores de rho.
# ------------------------------------------------------------------------
EPOCHS = 8
RHOS   = [0.0, 0.02, 0.05, 0.1, 0.2, 0.4, 0.8, 1.5]   # rho=0 equivale a SGD puro
history = {}   # rho -> dict com curvas

for rho in RHOS:
    model = make_model(input_shape, n_classes)
    opt = SAM(model.parameters(), torch.optim.SGD, rho=rho, lr=0.05, momentum=0.9)
    lam_curve, ep_axis = [], []
    for ep in range(EPOCHS):
        model.train()
        for x, y in train_loader:
            x, y = x.to(device), y.to(device)
            # passo 1: gradiente no ponto atual
            F.cross_entropy(model(x), y).backward()
            if rho > 0:
                opt.first_step()
                opt.zero_grad()
                # passo 2: gradiente no ponto perturbado
                F.cross_entropy(model(x), y).backward()
                opt.second_step()
            else:
                opt.base.step()                # rho=0 -> SGD comum
            opt.zero_grad()
        lam = power_iteration(model, hx, hy)
        lam_curve.append(lam); ep_axis.append(ep)
        print(f"rho={rho:<4} ep={ep}  lambda_max={lam:8.2f}")

    tr, te = accuracy(model, train_loader), accuracy(model, test_loader)
    history[rho] = dict(ep=ep_axis, lam=lam_curve,
                        lam_final=lam_curve[-1], gap=tr-te, train=tr, test=te)
    print(f"  rho={rho}: train={tr:.3f} test={te:.3f} gap={tr-te:.3f} lambda_max={lam_curve[-1]:.2f}\n")

# %% ----------------------------------------------------------------------
# Graficos
# ------------------------------------------------------------------------
# (1) lambda_max vs epoca, uma curva por rho
plt.figure(figsize=(7,4))
for rho in RHOS:
    h = history[rho]
    plt.plot(h["ep"], h["lam"], marker="o", label=f"rho={rho}")
plt.xlabel("epoca"); plt.ylabel(r"$\lambda_{\max}(H)$")
plt.title(f"Sharpness ao longo do treino ({DATASET})"); plt.legend(); plt.grid(True, alpha=.3)
plt.tight_layout(); plt.savefig(f"lambda_vs_epoch_{DATASET}.png", dpi=130); plt.show()

# (2) gap de generalizacao vs lambda_max (scatter sobre todos os rhos)
plt.figure(figsize=(6,4))
lams = [history[r]["lam_final"] for r in RHOS]
gaps = [history[r]["gap"]       for r in RHOS]
plt.scatter(lams, gaps, c=range(len(RHOS)), cmap="viridis", s=80)
for r in RHOS:
    plt.annotate(f"rho={r}", (history[r]["lam_final"], history[r]["gap"]),
                 textcoords="offset points", xytext=(5,5), fontsize=8)
plt.xlabel(r"$\lambda_{\max}(H)$ final"); plt.ylabel("gap (train acc - test acc)")
plt.title(f"Sharpness x generalizacao ({DATASET})"); plt.grid(True, alpha=.3)
plt.tight_layout(); plt.savefig(f"gap_vs_lambda_{DATASET}.png", dpi=130); plt.show()

# (3) lambda_max(rho) e gap(rho) em eixos gemeos
fig, ax1 = plt.subplots(figsize=(6,4))
ax1.plot(RHOS, lams, "o-", color="tab:blue"); ax1.set_xlabel("rho")
ax1.set_ylabel(r"$\lambda_{\max}(H)$", color="tab:blue"); ax1.tick_params(axis="y", labelcolor="tab:blue")
ax2 = ax1.twinx()
ax2.plot(RHOS, gaps, "s--", color="tab:red")
ax2.set_ylabel("gap de generalizacao", color="tab:red"); ax2.tick_params(axis="y", labelcolor="tab:red")
plt.title(rf"Efeito de $\rho$ sobre sharpness e generalizacao ({DATASET})")
plt.tight_layout(); plt.savefig(f"lambda_gap_vs_rho_{DATASET}.png", dpi=130); plt.show()
