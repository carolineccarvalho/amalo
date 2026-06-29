"""
Visualização da Geometria da Função de Perda
Baseado em: Li et al. (2018) "Visualizing the Loss Landscape of Neural Nets"

Técnica:
  - Parte dos parâmetros convergidos θ*
  - Define 1 ou 2 direções aleatórias δ (normalizadas por filtro)
  - Plota L(θ* + α·δ) ao longo dessas direções

NORMALIZAÇÃO POR FILTRO:
  - Escala cada direção aleatória para ter a mesma norma do peso correspondente
  - Remove o efeito da escala dos parâmetros, tornando a comparação entre modelos justa
  - Sem isso, a "largeza" do vale muda dependendo do tamanho dos pesos, não da curvatura real
"""

import copy
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn


# ============================================================
# UTILITÁRIOS
# ============================================================

def get_flat_params(model):
    """Pega todos os parâmetros do modelo como um vetor 1D."""
    return torch.cat([p.data.view(-1) for p in model.parameters()])


def set_flat_params(model, flat_params):
    """Seta os parâmetros do modelo a partir de um vetor 1D."""
    idx = 0
    for p in model.parameters():
        n = p.numel()
        p.data.copy_(flat_params[idx:idx + n].view(p.shape))
        idx += n


def filter_normalize_direction(direction, model):
    """
    Normalização por filtro (Li et al. 2018).
    Escala cada 'filtro' (linha de peso) da direção aleatória para que
    sua norma seja igual à norma do filtro correspondente no modelo.
    Isso torna a visualização independente da escala dos pesos.
    """
    normalized = []
    dir_idx = 0
    for p in model.parameters():
        n = p.numel()
        d_chunk = direction[dir_idx:dir_idx + n].view(p.shape)

        if p.dim() >= 2:
            for i in range(p.shape[0]):
                d_norm = d_chunk[i].norm()
                p_norm = p.data[i].norm()
                if d_norm > 1e-10:
                    d_chunk[i] = d_chunk[i] / d_norm * p_norm
        else:
            d_norm = d_chunk.norm()
            p_norm = p.data.norm()
            if d_norm > 1e-10:
                d_chunk = d_chunk / d_norm * p_norm

        normalized.append(d_chunk.view(-1))
        dir_idx += n

    return torch.cat(normalized)


def compute_loss(model, inputs, targets, criterion):
    """Calcula a loss do modelo sobre os dados fornecidos."""
    model.eval()
    with torch.no_grad():
        outputs = model(inputs)
        loss = criterion(outputs, targets)
    return loss.item()


# ============================================================
# CORTE 1D DA SUPERFÍCIE DE PERDA
# ============================================================

def plot_loss_1d(model, inputs, targets, criterion,
                 alphas=np.linspace(-1.0, 1.0, 51),
                 title="Geometria da Loss (Corte 1D)",
                 label=None, ax=None, color='steelblue',
                 use_filter_norm=True):
    """
    Plota L(θ* + α·δ) para α ∈ alphas.
    δ é uma direção aleatória com seed fixo (para comparação justa entre modelos).

    Parâmetros
    ----------
    model           : modelo PyTorch já treinado (θ*)
    inputs/targets  : dados para avaliar a loss
    alphas          : valores de α — controla o "zoom" no vale
    use_filter_norm : True recomendado para comparações entre modelos
    """
    original_params = get_flat_params(model).clone()
    n_params = original_params.numel()

    torch.manual_seed(0)  # seed fixo → mesma direção para todos os modelos
    direction = torch.randn(n_params)
    if use_filter_norm:
        direction = filter_normalize_direction(direction, model)

    losses = []
    for alpha in alphas:
        perturbed = original_params + alpha * direction
        set_flat_params(model, perturbed)
        losses.append(compute_loss(model, inputs, targets, criterion))

    set_flat_params(model, original_params)

    if ax is None:
        fig, ax = plt.subplots(figsize=(8, 4))

    label = label or "modelo"
    ax.plot(alphas, losses, label=label, color=color, linewidth=2)
    ax.axvline(x=0, color='gray', linestyle=':', linewidth=1, alpha=0.7)
    ax.set_xlabel("α (perturbação nos pesos)", fontsize=11)
    ax.set_ylabel("Loss", fontsize=11)
    ax.set_title(title, fontsize=13)
    ax.legend()
    ax.grid(True, alpha=0.3)

    return losses


def compare_models_1d(models_dict, inputs, targets, criterion,
                       alphas=np.linspace(-1.0, 1.0, 71),
                       title="Comparação de Geometria da Loss"):
    """
    Compara a geometria 1D de múltiplos modelos no mesmo gráfico.

    Exemplo de uso:
        compare_models_1d(
            {'Adam (agudo)': modelo_adam, 'SAM (plano)': modelo_sam},
            X_train_tensor, y_train_tensor,
            nn.BCEWithLogitsLoss()
        )
    """
    colors = ['#D85A30', '#3A37DD', '#2CA02C', '#9467BD', '#8C564B']
    fig, ax = plt.subplots(figsize=(9, 5))

    for (label, model), color in zip(models_dict.items(), colors):
        plot_loss_1d(model, inputs, targets, criterion,
                     alphas=alphas, label=label, ax=ax, color=color)

    ax.set_title(title, fontsize=13)
    fig.tight_layout()
    plt.show()
    return fig


# ============================================================
# SUPERFÍCIE 2D DA LOSS (MAPA DE CALOR + 3D)
# ============================================================

def compute_loss_landscape_2d(model, inputs, targets, criterion,
                               alphas=np.linspace(-1.0, 1.0, 25),
                               betas=np.linspace(-1.0, 1.0, 25),
                               use_filter_norm=True):
    """
    Computa a grade L(θ* + α·δ₁ + β·δ₂) para duas direções aleatórias.
    Retorna a matriz de losses com shape (len(betas), len(alphas)).

    25x25 = 625 avaliações — razoável para datasets pequenos.
    Aumente para 40x40 se quiser mais resolução (e tiver paciência).
    """
    original_params = get_flat_params(model).clone()
    n_params = original_params.numel()

    torch.manual_seed(0)
    dir1 = torch.randn(n_params)
    dir2 = torch.randn(n_params)

    if use_filter_norm:
        dir1 = filter_normalize_direction(dir1, model)
        dir2 = filter_normalize_direction(dir2, model)

    loss_grid = np.zeros((len(betas), len(alphas)))
    total = len(alphas) * len(betas)
    count = 0

    for i, beta in enumerate(betas):
        for j, alpha in enumerate(alphas):
            perturbed = original_params + alpha * dir1 + beta * dir2
            set_flat_params(model, perturbed)
            loss_grid[i, j] = compute_loss(model, inputs, targets, criterion)
            count += 1
            if count % 100 == 0:
                print(f"  Progresso: {count}/{total}")

    set_flat_params(model, original_params)
    return loss_grid


def plot_landscape_2d(loss_grid, alphas, betas,
                       title="Superfície de Perda 2D",
                       vmax_percentile=95):
    """
    Plota o mapa de calor 2D e a superfície 3D da função de perda.

    vmax_percentile: trunca a escala de cores — evita que picos extremos
    de um mínimo muito agudo dominem e "achatem" a visualização do vale.
    """
    fig = plt.figure(figsize=(13, 5))

    vmin = loss_grid.min()
    vmax = np.percentile(loss_grid, vmax_percentile)

    # --- Heatmap 2D ---
    ax1 = fig.add_subplot(1, 2, 1)
    im = ax1.contourf(alphas, betas, loss_grid,
                       levels=30, cmap='RdYlBu_r', vmin=vmin, vmax=vmax)
    ax1.contour(alphas, betas, loss_grid,
                levels=15, colors='black', linewidths=0.4, alpha=0.3)
    ax1.scatter([0], [0], color='white', s=80, zorder=5,
                edgecolors='black', linewidths=1.5, label='θ* (mínimo)')
    ax1.set_xlabel("α (direção 1)", fontsize=11)
    ax1.set_ylabel("β (direção 2)", fontsize=11)
    ax1.set_title(f"{title}\n(Mapa de Calor 2D)", fontsize=12)
    ax1.legend(fontsize=9)
    plt.colorbar(im, ax=ax1, label='Loss')

    # --- Superfície 3D ---
    ax2 = fig.add_subplot(1, 2, 2, projection='3d')
    A, B = np.meshgrid(alphas, betas)
    Z = np.clip(loss_grid, vmin, vmax)
    ax2.plot_surface(A, B, Z, cmap='RdYlBu_r',
                     linewidth=0, antialiased=True, alpha=0.9)
    ax2.set_xlabel("α", fontsize=9)
    ax2.set_ylabel("β", fontsize=9)
    ax2.set_zlabel("Loss", fontsize=9)
    ax2.set_title(f"{title}\n(Superfície 3D)", fontsize=12)
    ax2.view_init(elev=30, azim=-60)

    fig.tight_layout()
    plt.show()
    return fig


# ============================================================
# RASTREAMENTO DO λ_max DURANTE O TREINAMENTO
# ============================================================

def calcular_autovalor_dominante_full(inputs, targets, model, criterion,
                                       num_iterations=10, seed=None):
    """
    Calcula o lambda_max da Hessiana via Pearlmutter + Power Method.
    Versão interna usada por train_with_sharpness_tracking (retorna escalar).
    """
    model.eval()
    if seed is not None:
        torch.manual_seed(seed)
    num_params = sum(p.numel() for p in model.parameters())
    v = torch.randn(num_params, device=inputs.device)
    v = v / torch.norm(v)
    eigenvalue = 0.0

    for _ in range(num_iterations):
        model.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, targets)
        grads = torch.autograd.grad(loss, model.parameters(), create_graph=True)
        flat_grads = torch.cat([g.contiguous().view(-1) for g in grads])
        grad_v_prod = torch.dot(flat_grads, v)
        hvp = torch.autograd.grad(grad_v_prod, model.parameters())
        flat_hvp = torch.cat([h.contiguous().view(-1) for h in hvp])
        eigenvalue = torch.dot(v, flat_hvp).item()
        hvp_norm = torch.norm(flat_hvp)
        if hvp_norm.item() < 1e-12:
            break
        v = flat_hvp / hvp_norm

    return eigenvalue


def plot_sharpness_over_time(histories_dict,
                              title="λ_max durante o Treinamento"):
    """
    Plota a evolução do λ_max ao longo das épocas para múltiplos modelos.

    histories_dict: {'Adam': historico_adam, 'SAM': historico_sam}
    Os históricos precisam ter o campo 'lambda_max_history': lista de (epoch, λ_max),
    gerado por train_with_sharpness_tracking() abaixo.
    """
    colors = ['#D85A30', '#3A37DD', '#2CA02C']
    fig, ax = plt.subplots(figsize=(8, 4))

    for (label, history), color in zip(histories_dict.items(), colors):
        if 'lambda_max_history' not in history or not history['lambda_max_history']:
            print(f"  Aviso: '{label}' não tem lambda_max_history.")
            continue
        epochs_tracked = [ep for ep, _ in history['lambda_max_history']]
        lambdas = [lam for _, lam in history['lambda_max_history']]
        ax.plot(epochs_tracked, lambdas, label=label, color=color,
                marker='o', markersize=4, linewidth=2)

    ax.set_xlabel("Época", fontsize=11)
    ax.set_ylabel("λ_max(H)", fontsize=11)
    ax.set_title(title, fontsize=13)
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    plt.show()
    return fig


def train_with_sharpness_tracking(model, train_loader, val_loader,
                                   optimizer, criterion,
                                   epochs=300, is_sam=False,
                                   patience=40, track_every=10,
                                   X_full=None, y_full=None,
                                   restore_best=True, lambda_seed=42):
    """
    Versão extendida do train_model que rastreia λ_max a cada `track_every` épocas.

    Permite visualizar como a curvatura evolui durante o treinamento —
    narrativa muito mais rica sobre o comportamento sharp vs flat.

    Parâmetros extras:
        track_every : calcular λ_max a cada N épocas
        X_full, y_full : dados completos de treino (tensores) para o cálculo do λ_max

    Retorna history com campo extra 'lambda_max_history': lista de (epoch, λ_max).
    """
    history = {
        'train_loss': [], 'val_loss': [],
        'train_acc': [], 'val_acc': [],
        'lambda_max_history': []
    }
    best_state = copy.deepcopy(model.state_dict())
    best_val_loss = float('inf')
    epochs_without_improvement = 0

    for epoch in range(epochs):
        model.train()
        train_loss, train_correct, train_total = 0.0, 0, 0

        for inputs, targets in train_loader:
            if is_sam:
                outputs = model(inputs)
                loss = criterion(outputs, targets)
                loss.backward()
                optimizer.first_step(zero_grad=True)
                outputs2 = model(inputs)
                loss2 = criterion(outputs2, targets)
                loss2.backward()
                optimizer.second_step(zero_grad=True)
                with torch.no_grad():
                    outputs_to_log = model(inputs)
                    loss_to_log = criterion(outputs_to_log, targets)
            else:
                optimizer.zero_grad()
                outputs = model(inputs)
                loss = criterion(outputs, targets)
                loss.backward()
                optimizer.step()
                loss_to_log, outputs_to_log = loss, outputs

            train_loss += loss_to_log.item()
            if outputs_to_log.dim() > 1 and outputs_to_log.shape[1] > 1:
                preds = outputs_to_log.argmax(dim=1)
            else:
                preds = (outputs_to_log.squeeze() > 0.0).long()
            train_correct += (preds == targets).sum().item()
            train_total += targets.size(0)

        model.eval()
        val_loss, val_correct, val_total = 0.0, 0, 0
        with torch.no_grad():
            for inputs, targets in val_loader:
                outputs = model(inputs)
                loss = criterion(outputs, targets)
                val_loss += loss.item()
                if outputs.dim() > 1 and outputs.shape[1] > 1:
                    preds = outputs.argmax(dim=1)
                else:
                    preds = (outputs.squeeze() > 0.0).long()
                val_correct += (preds == targets).sum().item()
                val_total += targets.size(0)

        avg_train_loss = train_loss / len(train_loader)
        avg_val_loss = val_loss / len(val_loader)
        history['train_loss'].append(avg_train_loss)
        history['val_loss'].append(avg_val_loss)
        history['train_acc'].append(train_correct / train_total)
        history['val_acc'].append(val_correct / val_total)

        # Rastreia λ_max periodicamente
        if X_full is not None and (epoch + 1) % track_every == 0:
            lam = calcular_autovalor_dominante_full(
                X_full, y_full, model, criterion, num_iterations=10, seed=lambda_seed
            )
            history['lambda_max_history'].append((epoch + 1, lam))
            print(f"  Época {epoch + 1}: λ_max = {lam:.4f}")

        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            best_state = copy.deepcopy(model.state_dict())
            epochs_without_improvement = 0
        else:
            epochs_without_improvement += 1
            if patience and epochs_without_improvement >= patience:
                print(f"  Early stopping na época {epoch + 1}.")
                break

    if restore_best:
        model.load_state_dict(best_state)
    history['best_val_loss'] = best_val_loss
    history['epochs_trained'] = len(history['train_loss'])
    history['restore_best'] = restore_best
    return history


# ============================================================
# EXPERIMENTO DE PERTURBAÇÃO DIRECIONAL
# ============================================================

def perturbation_experiment(model, inputs, targets, criterion,
                             epsilons=np.linspace(0, 0.2, 30),
                             n_trials=5,
                             title="Sensibilidade a Perturbações"):
    """
    Para cada ε em epsilons, perturba θ* por ε na:
      (a) direção do autovetor dominante — direção de máxima curvatura
      (b) direções aleatórias (n_trials, para média e desvio)

    Mede ΔL = L(θ* + ε·d) - L(θ*).

    Um mínimo AGUDO mostra ΔL grande na direção dominante.
    Um mínimo PLANO mostra ΔL pequeno em ambas.
    """
    original_params = get_flat_params(model).clone()
    base_loss = compute_loss(model, inputs, targets, criterion)
    n_params = original_params.numel()

    # Estima o autovetor dominante (20 iterações do Power Method)
    v = torch.randn(n_params)
    v = v / torch.norm(v)
    for _ in range(20):
        model.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, targets)
        grads = torch.autograd.grad(loss, model.parameters(), create_graph=True)
        flat_grads = torch.cat([g.contiguous().view(-1) for g in grads])
        grad_v = torch.dot(flat_grads, v)
        hvp = torch.autograd.grad(grad_v, model.parameters())
        flat_hvp = torch.cat([h.contiguous().view(-1) for h in hvp])
        v = flat_hvp / torch.norm(flat_hvp)
    dominant_dir = v.detach()
    set_flat_params(model, original_params)

    delta_dominant = []
    delta_random_mean = []
    delta_random_std = []

    for eps in epsilons:
        # Direção dominante
        set_flat_params(model, original_params + eps * dominant_dir)
        delta_dominant.append(compute_loss(model, inputs, targets, criterion) - base_loss)

        # Direções aleatórias
        rand_deltas = []
        for _ in range(n_trials):
            rand_dir = torch.randn(n_params)
            rand_dir = rand_dir / torch.norm(rand_dir)
            set_flat_params(model, original_params + eps * rand_dir)
            rand_deltas.append(compute_loss(model, inputs, targets, criterion) - base_loss)
        delta_random_mean.append(np.mean(rand_deltas))
        delta_random_std.append(np.std(rand_deltas))

        set_flat_params(model, original_params)

    set_flat_params(model, original_params)

    fig, ax = plt.subplots(figsize=(8, 4))
    ax.plot(epsilons, delta_dominant,
            label='Direção dominante (λ_max)', color='#D85A30', linewidth=2)
    ax.plot(epsilons, delta_random_mean,
            label='Direção aleatória (média)', color='#3A37DD', linewidth=2)
    ax.fill_between(
        epsilons,
        np.array(delta_random_mean) - np.array(delta_random_std),
        np.array(delta_random_mean) + np.array(delta_random_std),
        alpha=0.2, color='#3A37DD', label='±1 std (aleatório)'
    )
    ax.axhline(y=0, color='gray', linestyle=':', linewidth=1)
    ax.set_xlabel("ε (magnitude da perturbação)", fontsize=11)
    ax.set_ylabel("ΔLoss = L(θ* + ε·d) − L(θ*)", fontsize=11)
    ax.set_title(title, fontsize=13)
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    plt.show()
    return fig
