"""
Estudo forte: minimos agudos/planos via lambda_max da Hessiana.

O script separa duas perguntas que o notebook misturava:
1. O otimizador aprendeu?  (train/test loss e acc)
2. A solucao aprendida e plana?  (lambda_max e corte na direcao dominante)

Fluxo recomendado:
  python strong_sharpness_study.py --dataset cifar10 --epochs 100 --seeds 0 1 2 3 4

Para um ensaio rapido:
  python strong_sharpness_study.py --dataset cifar10 --epochs 5 --n-train 2000 --n-test 500 --n-hess 128 --seeds 0
"""

from __future__ import annotations

import argparse
import copy
import csv
import json
import math
import random
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import torchvision.transforms as T
from torch.utils.data import DataLoader, Dataset, Subset, TensorDataset


CIFAR10_MEAN = (0.4914, 0.4822, 0.4465)
CIFAR10_STD = (0.2470, 0.2435, 0.2616)
CIFAR100_MEAN = (0.5071, 0.4865, 0.4409)
CIFAR100_STD = (0.2673, 0.2564, 0.2762)


@dataclass(frozen=True)
class OptimConfig:
    name: str
    family: str
    lr: float
    weight_decay: float
    momentum: float = 0.9
    rho: float = 0.05


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Comparacao justa de Adam, SGD e SAM para sharpness/generalizacao."
    )
    parser.add_argument("--dataset", choices=["cifar10", "cifar100"], default="cifar10")
    parser.add_argument("--data-root", default="./data")
    parser.add_argument("--out-dir", default="./figuras/strong_sharpness")
    parser.add_argument("--model", choices=["small_cnn", "resnet18"], default="small_cnn")
    parser.add_argument("--n-train", type=int, default=8000)
    parser.add_argument("--n-test", type=int, default=2000)
    parser.add_argument("--val-frac", type=float, default=0.15)
    parser.add_argument("--n-hess", type=int, default=512)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--eval-batch-size", type=int, default=512)
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--lambda-iters", type=int, default=20, help="Passos de Lanczos para estimar lambda_max.")
    parser.add_argument("--track-every", type=int, default=10)
    parser.add_argument("--cut-radius", type=float, default=0.35)
    parser.add_argument("--cut-points", type=int, default=41)
    parser.add_argument(
        "--match-train-acc-window",
        type=float,
        default=0.05,
        help="Resumo matched mantem runs com train_acc a ate esta distancia da melhor train_acc da seed.",
    )
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument(
        "--quick",
        action="store_true",
        help="Reduz o custo para verificar o pipeline rapidamente.",
    )
    return parser.parse_args()


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def make_dataset(dataset: str, root: str, train: bool) -> Dataset:
    if dataset == "cifar10":
        transform = T.Compose([T.ToTensor(), T.Normalize(CIFAR10_MEAN, CIFAR10_STD)])
        return torchvision.datasets.CIFAR10(root=root, train=train, download=True, transform=transform)
    transform = T.Compose([T.ToTensor(), T.Normalize(CIFAR100_MEAN, CIFAR100_STD)])
    return torchvision.datasets.CIFAR100(root=root, train=train, download=True, transform=transform)


def random_subset_indices(n_total: int, n_keep: int, seed: int) -> list[int]:
    generator = torch.Generator().manual_seed(seed)
    perm = torch.randperm(n_total, generator=generator).tolist()
    return perm[: min(n_keep, n_total)]


def make_loaders(args: argparse.Namespace, seed: int):
    train_full = make_dataset(args.dataset, args.data_root, train=True)
    test_full = make_dataset(args.dataset, args.data_root, train=False)

    train_val_indices = random_subset_indices(len(train_full), args.n_train, seed)
    n_val = max(1, int(round(len(train_val_indices) * args.val_frac)))
    generator = torch.Generator().manual_seed(seed + 10_000)
    shuffled = torch.tensor(train_val_indices)[torch.randperm(len(train_val_indices), generator=generator)].tolist()
    val_indices = shuffled[:n_val]
    train_indices = shuffled[n_val:]
    test_indices = random_subset_indices(len(test_full), args.n_test, seed + 20_000)

    train_ds = Subset(train_full, train_indices)
    val_ds = Subset(train_full, val_indices)
    test_ds = Subset(test_full, test_indices)

    train_loader = DataLoader(
        train_ds,
        batch_size=args.batch_size,
        shuffle=True,
        generator=torch.Generator().manual_seed(seed + 30_000),
        num_workers=args.num_workers,
    )
    val_loader = DataLoader(val_ds, batch_size=args.eval_batch_size, shuffle=False, num_workers=args.num_workers)
    test_loader = DataLoader(test_ds, batch_size=args.eval_batch_size, shuffle=False, num_workers=args.num_workers)

    hess_count = min(args.n_hess, len(train_ds))
    xs, ys = [], []
    for i in range(hess_count):
        x, y = train_ds[i]
        xs.append(x)
        ys.append(y)
    hx = torch.stack(xs).to(args.device)
    hy = torch.tensor(ys, dtype=torch.long, device=args.device)
    return train_loader, val_loader, test_loader, hx, hy


class SmallCNN(nn.Module):
    def __init__(self, num_classes: int):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 64, 3, padding=1),
            nn.GroupNorm(8, 64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Conv2d(64, 128, 3, padding=1),
            nn.GroupNorm(16, 128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Conv2d(128, 256, 3, padding=1),
            nn.GroupNorm(32, 256),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d(1),
        )
        self.classifier = nn.Linear(256, num_classes)

    def forward(self, x):
        return self.classifier(torch.flatten(self.features(x), 1))


def make_model(name: str, num_classes: int) -> nn.Module:
    if name == "small_cnn":
        return SmallCNN(num_classes)
    model = torchvision.models.resnet18(weights=None, num_classes=num_classes)
    model.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
    model.maxpool = nn.Identity()
    return model


class SAM(torch.optim.Optimizer):
    def __init__(self, params, base_optimizer, rho=0.05, **kwargs):
        defaults = dict(rho=rho, **kwargs)
        super().__init__(params, defaults)
        self.base_optimizer = base_optimizer(self.param_groups, **kwargs)
        self.param_groups = self.base_optimizer.param_groups

    @torch.no_grad()
    def first_step(self, zero_grad=False):
        grad_norm = self._grad_norm()
        for group in self.param_groups:
            scale = group["rho"] / (grad_norm + 1e-12)
            for p in group["params"]:
                if p.grad is None:
                    continue
                e_w = p.grad * scale
                p.add_(e_w)
                self.state[p]["e_w"] = e_w
        if zero_grad:
            self.zero_grad()

    @torch.no_grad()
    def second_step(self, zero_grad=False):
        for group in self.param_groups:
            for p in group["params"]:
                if p.grad is None:
                    continue
                p.sub_(self.state[p]["e_w"])
        self.base_optimizer.step()
        if zero_grad:
            self.zero_grad()

    def _grad_norm(self):
        norms = [
            p.grad.norm(p=2)
            for group in self.param_groups
            for p in group["params"]
            if p.grad is not None
        ]
        if not norms:
            return torch.tensor(0.0, device=self.param_groups[0]["params"][0].device)
        return torch.norm(torch.stack(norms), p=2)


def default_configs(dataset: str) -> list[OptimConfig]:
    # Configuracoes conservadoras: Adam usa LR menor que SGD.
    wd = 5e-4 if dataset.startswith("cifar") else 1e-4
    return [
        OptimConfig("SGD", "sgd", lr=0.05, weight_decay=wd),
        OptimConfig("Adam", "adam", lr=1e-3, weight_decay=wd),
        OptimConfig("SAM-SGD rho=0.05", "sam_sgd", lr=0.05, weight_decay=wd, rho=0.05),
        OptimConfig("SAM-Adam rho=0.05", "sam_adam", lr=1e-3, weight_decay=wd, rho=0.05),
    ]


def make_optimizer(config: OptimConfig, model: nn.Module):
    if config.family == "sgd":
        return torch.optim.SGD(
            model.parameters(), lr=config.lr, momentum=config.momentum, weight_decay=config.weight_decay
        )
    if config.family == "adam":
        return torch.optim.Adam(model.parameters(), lr=config.lr, weight_decay=config.weight_decay)
    if config.family == "sam_sgd":
        return SAM(
            model.parameters(),
            torch.optim.SGD,
            lr=config.lr,
            momentum=config.momentum,
            weight_decay=config.weight_decay,
            rho=config.rho,
        )
    if config.family == "sam_adam":
        return SAM(
            model.parameters(),
            torch.optim.Adam,
            lr=config.lr,
            weight_decay=config.weight_decay,
            rho=config.rho,
        )
    raise ValueError(f"Unknown optimizer family: {config.family}")


def is_sam(config: OptimConfig) -> bool:
    return config.family.startswith("sam_")


@torch.no_grad()
def evaluate(model: nn.Module, loader: DataLoader, device: str) -> dict[str, float]:
    model.eval()
    total_loss = 0.0
    total_correct = 0
    total = 0
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        logits = model(x)
        loss = F.cross_entropy(logits, y)
        total_loss += loss.item() * y.numel()
        total_correct += (logits.argmax(1) == y).sum().item()
        total += y.numel()
    return {"loss": total_loss / total, "acc": total_correct / total}


def train_one_epoch(model: nn.Module, loader: DataLoader, optimizer, config: OptimConfig, device: str) -> dict[str, float]:
    model.train()
    total_loss = 0.0
    total_correct = 0
    total = 0
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        if is_sam(config):
            logits = model(x)
            loss = F.cross_entropy(logits, y)
            loss.backward()
            optimizer.first_step(zero_grad=True)
            F.cross_entropy(model(x), y).backward()
            optimizer.second_step(zero_grad=True)
            with torch.no_grad():
                logits_log = model(x)
                loss_log = F.cross_entropy(logits_log, y)
        else:
            optimizer.zero_grad()
            logits_log = model(x)
            loss_log = F.cross_entropy(logits_log, y)
            loss_log.backward()
            optimizer.step()
        total_loss += loss_log.item() * y.numel()
        total_correct += (logits_log.argmax(1) == y).sum().item()
        total += y.numel()
    return {"loss": total_loss / total, "acc": total_correct / total}


def flat_params(model: nn.Module) -> torch.Tensor:
    return torch.cat([p.detach().reshape(-1) for p in model.parameters() if p.requires_grad])


def set_flat_params(model: nn.Module, vector: torch.Tensor) -> None:
    offset = 0
    with torch.no_grad():
        for p in model.parameters():
            if not p.requires_grad:
                continue
            n = p.numel()
            p.copy_(vector[offset : offset + n].view_as(p))
            offset += n


def split_like_params(vector: torch.Tensor, params: Iterable[torch.Tensor]) -> list[torch.Tensor]:
    chunks = []
    offset = 0
    for p in params:
        n = p.numel()
        chunks.append(vector[offset : offset + n].view_as(p))
        offset += n
    return chunks


def hessian_vector_product(
    model: nn.Module,
    inputs: torch.Tensor,
    targets: torch.Tensor,
    params: list[torch.Tensor],
    vector: torch.Tensor,
) -> torch.Tensor:
    model.zero_grad(set_to_none=True)
    logits = model(inputs)
    loss = F.cross_entropy(logits, targets)
    grads = torch.autograd.grad(loss, params, create_graph=True)
    v_chunks = split_like_params(vector, params)
    grad_dot_v = sum((g * vc).sum() for g, vc in zip(grads, v_chunks))
    hvp = torch.autograd.grad(grad_dot_v, params)
    return torch.cat([h.contiguous().reshape(-1) for h in hvp]).detach()


def lambda_max_and_vector(
    model: nn.Module, inputs: torch.Tensor, targets: torch.Tensor, n_iter: int, seed: int
) -> tuple[float, torch.Tensor]:
    """Maior autovalor algebrico via Lanczos.

    Power method simples pode convergir para o autovalor de maior magnitude,
    inclusive negativo. Para falar de minimo agudo/plano, queremos o maior
    autovalor positivo/algebrico da Hessiana local.
    """
    model.eval()
    params = [p for p in model.parameters() if p.requires_grad]
    num_params = sum(p.numel() for p in params)
    generator = torch.Generator(device=inputs.device).manual_seed(seed)
    q = torch.randn(num_params, generator=generator, device=inputs.device)
    q = q / (q.norm() + 1e-12)
    q_prev = torch.zeros_like(q)
    beta_prev = torch.tensor(0.0, device=inputs.device)
    basis = []
    alphas = []
    betas = []

    steps = max(2, min(n_iter, num_params))
    for _ in range(steps):
        basis.append(q)
        z = hessian_vector_product(model, inputs, targets, params, q)
        alpha = torch.dot(q, z)
        z = z - alpha * q - beta_prev * q_prev
        # Reortogonalizacao curta contra a base acumulada; reduz instabilidade
        # numerica em modelos pequenos sem complicar o codigo.
        for old_q in basis:
            z = z - torch.dot(old_q, z) * old_q
        beta = z.norm()
        alphas.append(alpha)
        if beta.item() < 1e-10:
            break
        betas.append(beta)
        q_prev = q
        q = (z / beta).detach()
        beta_prev = beta

    k = len(alphas)
    tri = torch.zeros((k, k), device=inputs.device)
    for i, alpha in enumerate(alphas):
        tri[i, i] = alpha
    for i, beta in enumerate(betas[: max(0, k - 1)]):
        tri[i, i + 1] = beta
        tri[i + 1, i] = beta
    eigvals, eigvecs = torch.linalg.eigh(tri.cpu())
    top_idx = int(torch.argmax(eigvals).item())
    coeffs = eigvecs[:, top_idx].to(inputs.device)
    top_vec = torch.zeros(num_params, device=inputs.device)
    for coeff, q_i in zip(coeffs, basis):
        top_vec = top_vec + coeff * q_i
    top_vec = top_vec / (top_vec.norm() + 1e-12)
    return float(eigvals[top_idx].item()), top_vec.detach()


def loss_cut_along_vector(
    model: nn.Module,
    inputs: torch.Tensor,
    targets: torch.Tensor,
    direction: torch.Tensor,
    alphas: np.ndarray,
) -> list[float]:
    original = flat_params(model).clone()
    losses = []
    model.eval()
    for alpha in alphas:
        set_flat_params(model, original + float(alpha) * direction)
        with torch.no_grad():
            losses.append(F.cross_entropy(model(inputs), targets).item())
    set_flat_params(model, original)
    return losses


def train_run(
    args: argparse.Namespace,
    config: OptimConfig,
    seed: int,
    train_loader: DataLoader,
    val_loader: DataLoader,
    test_loader: DataLoader,
    hx: torch.Tensor,
    hy: torch.Tensor,
    num_classes: int,
):
    set_seed(seed)
    model = make_model(args.model, num_classes).to(args.device)
    optimizer = make_optimizer(config, model)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer.base_optimizer if is_sam(config) else optimizer, args.epochs)
    best_val_loss = float("inf")
    best_state = copy.deepcopy(model.state_dict())
    history = []
    lambda_history = []

    for epoch in range(1, args.epochs + 1):
        tr = train_one_epoch(model, train_loader, optimizer, config, args.device)
        val = evaluate(model, val_loader, args.device)
        scheduler.step()
        if val["loss"] < best_val_loss:
            best_val_loss = val["loss"]
            best_state = copy.deepcopy(model.state_dict())
        row = {
            "epoch": epoch,
            "train_loss": tr["loss"],
            "train_acc": tr["acc"],
            "val_loss": val["loss"],
            "val_acc": val["acc"],
        }
        history.append(row)
        if args.track_every and (epoch % args.track_every == 0 or epoch == args.epochs):
            lam, _ = lambda_max_and_vector(model, hx, hy, args.lambda_iters, seed + epoch)
            lambda_history.append({"epoch": epoch, "lambda_max": lam})
            print(f"{config.name:<18} seed={seed:<2} epoch={epoch:<4} val_acc={val['acc']:.3f} lambda={lam:.4f}")

    model.load_state_dict(best_state)
    train_metrics = evaluate(model, train_loader, args.device)
    val_metrics = evaluate(model, val_loader, args.device)
    test_metrics = evaluate(model, test_loader, args.device)
    lam_final, eigvec = lambda_max_and_vector(model, hx, hy, args.lambda_iters, seed + 999)
    result = {
        "optimizer": config.name,
        "family": config.family,
        "seed": seed,
        "lr": config.lr,
        "weight_decay": config.weight_decay,
        "momentum": config.momentum,
        "rho": config.rho if is_sam(config) else 0.0,
        "best_val_loss": best_val_loss,
        "train_loss": train_metrics["loss"],
        "train_acc": train_metrics["acc"],
        "val_loss": val_metrics["loss"],
        "val_acc": val_metrics["acc"],
        "test_loss": test_metrics["loss"],
        "test_acc": test_metrics["acc"],
        "gap_loss": test_metrics["loss"] - train_metrics["loss"],
        "gap_acc": train_metrics["acc"] - test_metrics["acc"],
        "lambda_max": lam_final,
    }
    return model, result, history, lambda_history, eigvec


def write_csv(path: Path, rows: list[dict]) -> None:
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def summarize(rows: list[dict]) -> list[dict]:
    summary = []
    for optimizer in sorted({r["optimizer"] for r in rows}):
        group = [r for r in rows if r["optimizer"] == optimizer]
        out = {"optimizer": optimizer, "n": len(group)}
        for metric in ["train_acc", "test_acc", "gap_acc", "gap_loss", "lambda_max", "test_loss"]:
            vals = np.array([float(r[metric]) for r in group])
            out[f"{metric}_mean"] = vals.mean()
            out[f"{metric}_std"] = vals.std(ddof=1) if len(vals) > 1 else 0.0
        summary.append(out)
    return summary


def filter_matched_train_accuracy(rows: list[dict], window: float) -> list[dict]:
    """Mantem apenas solucoes que aprenderam em nivel parecido dentro de cada seed."""
    matched = []
    for seed in sorted({int(r["seed"]) for r in rows}):
        group = [r for r in rows if int(r["seed"]) == seed]
        best_train_acc = max(float(r["train_acc"]) for r in group)
        threshold = best_train_acc - window
        for row in group:
            if float(row["train_acc"]) >= threshold:
                kept = dict(row)
                kept["matched_train_acc_threshold"] = threshold
                matched.append(kept)
    return matched


def save_plots(out_dir: Path, rows: list[dict], lambda_rows: list[dict], cut_rows: list[dict]) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)

    plt.figure(figsize=(7, 4.5))
    for optimizer in sorted({r["optimizer"] for r in lambda_rows}):
        group = [r for r in lambda_rows if r["optimizer"] == optimizer]
        epochs = sorted({int(r["epoch"]) for r in group})
        means = []
        stds = []
        for ep in epochs:
            vals = np.array([float(r["lambda_max"]) for r in group if int(r["epoch"]) == ep])
            means.append(vals.mean())
            stds.append(vals.std(ddof=1) if len(vals) > 1 else 0.0)
        plt.errorbar(epochs, means, yerr=stds, marker="o", capsize=3, label=optimizer)
    plt.xlabel("epoca")
    plt.ylabel(r"$\lambda_{\max}(H)$")
    plt.title("Evolucao da sharpness")
    plt.grid(True, alpha=0.3)
    plt.legend(fontsize=8)
    plt.tight_layout()
    plt.savefig(out_dir / "lambda_por_epoca.png", dpi=150)
    plt.close()

    plt.figure(figsize=(6, 4.5))
    for optimizer in sorted({r["optimizer"] for r in rows}):
        group = [r for r in rows if r["optimizer"] == optimizer]
        plt.scatter(
            [float(r["lambda_max"]) for r in group],
            [float(r["gap_loss"]) for r in group],
            label=optimizer,
            s=55,
        )
    plt.xlabel(r"$\lambda_{\max}(H)$")
    plt.ylabel("gap loss = test_loss - train_loss")
    plt.title("Sharpness x generalizacao")
    plt.grid(True, alpha=0.3)
    plt.legend(fontsize=8)
    plt.tight_layout()
    plt.savefig(out_dir / "lambda_vs_gap_loss.png", dpi=150)
    plt.close()

    if cut_rows:
        plt.figure(figsize=(7, 4.5))
        for optimizer in sorted({r["optimizer"] for r in cut_rows}):
            group = sorted([r for r in cut_rows if r["optimizer"] == optimizer], key=lambda r: float(r["alpha"]))
            plt.plot(
                [float(r["alpha"]) for r in group],
                [float(r["loss"]) for r in group],
                marker="o",
                markersize=3,
                label=optimizer,
            )
        plt.xlabel(r"$\alpha$ na direcao dominante")
        plt.ylabel("loss")
        plt.title(r"Corte $L(\theta + \alpha v_{max})$")
        plt.grid(True, alpha=0.3)
        plt.legend(fontsize=8)
        plt.tight_layout()
        plt.savefig(out_dir / "corte_autovetor_dominante.png", dpi=150)
        plt.close()


def main() -> None:
    args = parse_args()
    if args.quick:
        args.epochs = min(args.epochs, 5)
        args.n_train = min(args.n_train, 2000)
        args.n_test = min(args.n_test, 500)
        args.n_hess = min(args.n_hess, 128)
        args.seeds = args.seeds[:1]
        args.lambda_iters = min(args.lambda_iters, 8)
        args.track_every = max(1, min(args.track_every, args.epochs))

    out_dir = Path(args.out_dir) / f"{args.dataset}_{args.model}"
    out_dir.mkdir(parents=True, exist_ok=True)
    with (out_dir / "run_config.json").open("w", encoding="utf-8") as f:
        json.dump(vars(args), f, indent=2)

    num_classes = 10 if args.dataset == "cifar10" else 100
    configs = default_configs(args.dataset)
    all_rows = []
    all_hist_rows = []
    all_lambda_rows = []
    cut_rows = []

    print(f"device={args.device} dataset={args.dataset} model={args.model}")
    print(f"configs={[asdict(c) for c in configs]}")

    for seed in args.seeds:
        train_loader, val_loader, test_loader, hx, hy = make_loaders(args, seed)
        representative = {}
        for config in configs:
            model, result, history, lambda_history, eigvec = train_run(
                args, config, seed, train_loader, val_loader, test_loader, hx, hy, num_classes
            )
            all_rows.append(result)
            for row in history:
                all_hist_rows.append({"optimizer": config.name, "seed": seed, **row})
            for row in lambda_history:
                all_lambda_rows.append({"optimizer": config.name, "seed": seed, **row})
            if seed == args.seeds[0]:
                representative[config.name] = (model, eigvec, hx, hy)
            print(
                f"FINAL {config.name:<18} seed={seed:<2} "
                f"train_acc={result['train_acc']:.3f} test_acc={result['test_acc']:.3f} "
                f"gap_loss={result['gap_loss']:+.4f} lambda={result['lambda_max']:.4f}"
            )

        if seed == args.seeds[0]:
            alphas = np.linspace(-args.cut_radius, args.cut_radius, args.cut_points)
            for opt_name, (model, eigvec, hx_cut, hy_cut) in representative.items():
                losses = loss_cut_along_vector(model, hx_cut, hy_cut, eigvec, alphas)
                base = losses[len(losses) // 2]
                for alpha, loss in zip(alphas, losses):
                    cut_rows.append(
                        {
                            "optimizer": opt_name,
                            "seed": seed,
                            "alpha": float(alpha),
                            "loss": float(loss),
                            "delta_loss": float(loss - base),
                        }
                    )

    write_csv(out_dir / "runs.csv", all_rows)
    write_csv(out_dir / "history.csv", all_hist_rows)
    write_csv(out_dir / "lambda_history.csv", all_lambda_rows)
    write_csv(out_dir / "dominant_eigen_cut.csv", cut_rows)
    summary = summarize(all_rows)
    write_csv(out_dir / "summary.csv", summary)
    matched_rows = filter_matched_train_accuracy(all_rows, args.match_train_acc_window)
    write_csv(out_dir / "matched_runs.csv", matched_rows)
    matched_summary = summarize(matched_rows)
    write_csv(out_dir / "matched_summary.csv", matched_summary)
    save_plots(out_dir, all_rows, all_lambda_rows, cut_rows)

    print("\nResumo:")
    for row in summary:
        print(
            f"{row['optimizer']:<18} "
            f"test_acc={row['test_acc_mean']:.3f}+-{row['test_acc_std']:.3f} "
            f"gap_loss={row['gap_loss_mean']:+.4f}+-{row['gap_loss_std']:.4f} "
            f"lambda={row['lambda_max_mean']:.4f}+-{row['lambda_max_std']:.4f}"
        )
    print(f"\nResumo matched por train_acc (janela={args.match_train_acc_window}):")
    for row in matched_summary:
        print(
            f"{row['optimizer']:<18} "
            f"n={row['n']:<2} "
            f"test_acc={row['test_acc_mean']:.3f}+-{row['test_acc_std']:.3f} "
            f"gap_loss={row['gap_loss_mean']:+.4f}+-{row['gap_loss_std']:.4f} "
            f"lambda={row['lambda_max_mean']:.4f}+-{row['lambda_max_std']:.4f}"
        )
    print(f"\nArquivos salvos em: {out_dir}")


if __name__ == "__main__":
    main()
