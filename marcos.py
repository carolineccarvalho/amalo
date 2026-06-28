python 
# %% ----------------------------------------------------------------------
# Experimento: sharpness (lambda_max) vs generalizacao sob SAM, variando rho
# FashionMNIST + MLP. HVP por Pearlmutter (reverse-over-reverse) + metodo da potencia.
# ------------------------------------------------------------------------
import math, copy
import torch, torch.nn as nn, torch.nn.functional as F
from torch.utils.data import DataLoader, Subset
import torchvision
import torchvision.transforms as T
import matplotlib.pyplot as plt

device = "cuda" if torch.cuda.is_available() else "cpu"
torch.manual_seed(0)
print("device:", device)

# ---- dados -------------------------------------------------------------
tf = T.Compose([T.ToTensor(), T.Normalize((0.286,), (0.353,))])
train_full = torchvision.datasets.FashionMNIST(root="./data", train=True,  download=True, transform=tf)
test_full  = torchvision.datasets.FashionMNIST(root="./data", train=False, download=True, transform=tf)

# subconjunto para rodar rapido no Colab (aumente se tiver GPU/tempo)
N_TRAIN, N_TEST = 16000, 4000
train_ds = Subset(train_full, range(N_TRAIN))
test_ds  = Subset(test_full,  range(N_TEST))

train_loader = DataLoader(train_ds, batch_size=128, shuffle=True)
test_loader  = DataLoader(test_ds,  batch_size=512, shuffle=False)

# batch fixo (grande) para estimar a Hessiana sempre no mesmo ponto
hx = torch.stack([train_ds[i][0] for i in range(1000)]).to(device)
hy = torch.tensor([train_ds[i][1] for i in range(1000)]).to(device)

# ---- modelo: MLP ~270k parametros --------------------------------------
def make_model():
    return nn.Sequential(
        nn.Flatten(),
        nn.Linear(28*28, 256), nn.ReLU(),
        nn.Linear(256, 256),   nn.ReLU(),
        nn.Linear(256, 10),
    ).to(device)

n_params = sum(p.numel() for p in make_model().parameters())
print(f"n_params = {n_params}  (Hessiana densa teria {n_params**2:.2e} entradas)")

# ---- SAM ---------------------------------------------------------------
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

# ---- avaliacao ---------------------------------------------------------
@torch.no_grad()
def accuracy(model, loader):
    model.eval(); correct = total = 0
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        correct += (model(x).argmax(1) == y).sum().item(); total += y.numel()
    return correct / total

# %% ----------------------------------------------------------------------
# Treino: para cada rho, treina e registra lambda_max por epoca
# ------------------------------------------------------------------------
EPOCHS = 8
RHOS   = [0.0, 0.02, 0.05, 0.1, 0.2, 0.4, 0.8, 1.5]   # rho=0 equivale a SGD puro
history = {}   # rho -> dict com curvas

for rho in RHOS:
    model = make_model()
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
plt.title("Sharpness ao longo do treino"); plt.legend(); plt.grid(True, alpha=.3)
plt.tight_layout(); plt.savefig("lambda_vs_epoch.png", dpi=130); plt.show()

# (2) gap de generalizacao vs lambda_max (scatter sobre todos os rhos)
plt.figure(figsize=(6,4))
lams = [history[r]["lam_final"] for r in RHOS]
gaps = [history[r]["gap"]       for r in RHOS]
plt.scatter(lams, gaps, c=range(len(RHOS)), cmap="viridis", s=80)
for r in RHOS:
    plt.annotate(f"rho={r}", (history[r]["lam_final"], history[r]["gap"]),
                 textcoords="offset points", xytext=(5,5), fontsize=8)
plt.xlabel(r"$\lambda_{\max}(H)$ final"); plt.ylabel("gap (train acc - test acc)")
plt.title("Sharpness x generalizacao"); plt.grid(True, alpha=.3)
plt.tight_layout(); plt.savefig("gap_vs_lambda.png", dpi=130); plt.show()

# (3) lambda_max(rho) e gap(rho) em eixos gemeos
fig, ax1 = plt.subplots(figsize=(6,4))
ax1.plot(RHOS, lams, "o-", color="tab:blue"); ax1.set_xlabel("rho")
ax1.set_ylabel(r"$\lambda_{\max}(H)$", color="tab:blue"); ax1.tick_params(axis="y", labelcolor="tab:blue")
ax2 = ax1.twinx()
ax2.plot(RHOS, gaps, "s--", color="tab:red")
ax2.set_ylabel("gap de generalizacao", color="tab:red"); ax2.tick_params(axis="y", labelcolor="tab:red")
plt.title(r"Efeito de $\rho$ sobre sharpness e generalizacao")
plt.tight_layout(); plt.savefig("lambda_gap_vs_rho.png", dpi=130); plt.show()