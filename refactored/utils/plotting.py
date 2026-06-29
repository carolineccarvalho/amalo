"""Funções de plotagem usadas pelos experimentos.

Cores fixas por otimizador (CORES) para que Adam/SGD/SAM tenham a mesma cor em
todos os gráficos, facilitando a comparação visual entre figuras.
"""

import os

import matplotlib.pyplot as plt
import numpy as np

CORES = {'Adam': '#3A37DD', 'SGD': '#2ca02c', 'SAM': '#D85A30'}


def save_fig(filename):
    os.makedirs('figuras', exist_ok=True)
    plt.savefig(f'figuras/{filename}', dpi=150, bbox_inches='tight')


# ============================================================
# Curvas de aprendizado, métricas e sharpness por otimizador
# ============================================================

def plot_comparacao(historicos, title=''):
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    for label, h in historicos.items():
        cor = CORES.get(label, 'gray')
        ep = range(1, len(h['train_loss']) + 1)
        axes[0].plot(ep, h['train_loss'], color=cor, label=f'{label} treino')
        axes[0].plot(ep, h['val_loss'],   color=cor, ls='--', alpha=0.6, label=f'{label} val')
        axes[1].plot(ep, h['train_acc'],  color=cor, label=f'{label} treino')
        axes[1].plot(ep, h['val_acc'],    color=cor, ls='--', alpha=0.6, label=f'{label} val')
    for ax, ylabel in zip(axes, ['loss', 'acurácia']):
        ax.set_xlabel('época'); ax.set_ylabel(ylabel)
        ax.legend(fontsize=7); ax.grid(True, alpha=.3)
    fig.suptitle(title); fig.tight_layout(); plt.show()


def plot_metricas_barras(resultados, title=''):
    labels = list(resultados.keys())
    tr_acc = [resultados[l]['train']['accuracy'] for l in labels]
    te_acc = [resultados[l]['test']['accuracy']  for l in labels]
    gaps   = [resultados[l]['gap']               for l in labels]
    x, w = np.arange(len(labels)), 0.3
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    axes[0].bar(x - w/2, tr_acc, w, label='treino', color='#3A37DD')
    axes[0].bar(x + w/2, te_acc, w, label='teste',  color='#D85A30')
    axes[0].set_xticks(x); axes[0].set_xticklabels(labels)
    axes[0].set_ylim(0, 1.05); axes[0].set_ylabel('acurácia')
    axes[0].legend(); axes[0].set_title('Acurácia treino vs. teste')
    cores_gap = [CORES.get(l, 'gray') for l in labels]
    axes[1].bar(x, gaps, color=cores_gap)
    axes[1].set_xticks(x); axes[1].set_xticklabels(labels)
    axes[1].set_ylabel('gap (treino − teste)'); axes[1].set_title('Gap de Generalização')
    axes[1].grid(True, alpha=.3, axis='y')
    fig.suptitle(title); fig.tight_layout(); plt.show()


def plot_lambda_bar(lambdas, title=''):
    labels = list(lambdas.keys())
    values = [lambdas[l] for l in labels]
    cores = [CORES.get(l, 'gray') for l in labels]
    plt.figure(figsize=(6, 4))
    bars = plt.bar(labels, values, color=cores)
    for bar, val in zip(bars, values):
        plt.text(bar.get_x() + bar.get_width()/2,
                 val + max(values)*0.015, f'{val:.2f}',
                 ha='center', va='bottom', fontsize=10)
    plt.ylabel(r'$\lambda_{\max}(H)$'); plt.title(title)
    plt.grid(True, alpha=.3, axis='y'); plt.tight_layout(); plt.show()


def plot_history(history, title=''):
    plot_comparacao({'modelo': history}, title=title)


# ============================================================
# Convergência do método da potência
# ============================================================

def plot_power_method(histories, title='Convergência do Método da Potência'):
    """`histories`: dict {label: lista de λ por iteração}."""
    plt.figure(figsize=(9, 5))
    for label, hist in histories.items():
        cor = CORES.get(label, 'gray')
        plt.plot(range(1, len(hist)+1), hist, marker='o', markersize=4,
                 label=f'{label} (final={hist[-1]:.3f})', color=cor)
    plt.xlabel('Iteração'); plt.ylabel(r'$\lambda_{\max}(H)$')
    plt.title(title)
    plt.legend(); plt.grid(True, ls='--', alpha=.6); plt.tight_layout(); plt.show()


# ============================================================
# Ablação de ρ — boxplots + eixos gêmeos
# ============================================================

def plot_rho_ablation_summary(df_rho, summary_rho, title=''):
    labels = list(summary_rho['nome'])
    positions = np.arange(len(labels))
    lambda_data = [df_rho.loc[df_rho['nome'] == l, 'lambda_max'].values for l in labels]
    gap_data    = [df_rho.loc[df_rho['nome'] == l, 'gap_loss'].values for l in labels]

    fig, axes = plt.subplots(1, 2, figsize=(14, 4.8))
    axes[0].boxplot(lambda_data, labels=labels, showmeans=True)
    axes[0].set_ylabel(r'$\lambda_{\max}(H)$'); axes[0].set_title('Sharpness por seed')
    axes[0].grid(True, alpha=.3, axis='y'); axes[0].tick_params(axis='x', rotation=25)
    axes[1].boxplot(gap_data, labels=labels, showmeans=True)
    axes[1].axhline(0, color='gray', ls=':', lw=1)
    axes[1].set_ylabel('gap loss (teste−treino)'); axes[1].set_title('Gap de Generalização por seed')
    axes[1].grid(True, alpha=.3, axis='y'); axes[1].tick_params(axis='x', rotation=25)
    fig.tight_layout(); plt.show()

    fig, ax1 = plt.subplots(figsize=(9, 4.8))
    ax1.errorbar(positions, summary_rho['lambda_mean'], yerr=summary_rho['lambda_std'],
                 marker='o', capsize=4, color='#D85A30', label=r'$\lambda_{\max}$')
    ax1.set_ylabel(r'$\lambda_{\max}(H)$', color='#D85A30')
    ax1.tick_params(axis='y', labelcolor='#D85A30')
    ax1.set_xticks(positions); ax1.set_xticklabels(labels, rotation=25, ha='right')
    ax1.grid(True, alpha=.3)
    ax2 = ax1.twinx()
    ax2.errorbar(positions, summary_rho['gap_loss_mean'], yerr=summary_rho['gap_loss_std'],
                 marker='s', capsize=4, color='#3A37DD', label='gap loss')
    ax2.set_ylabel('Gap loss', color='#3A37DD')
    ax2.tick_params(axis='y', labelcolor='#3A37DD')
    fig.suptitle(title)
    fig.tight_layout(); plt.show()


# ============================================================
# Landscape 2D — painel Adam vs SAM
# ============================================================

def plot_landscape_panel(grid_a, grid_b, name_a, name_b, alphas, betas,
                         suptitle='', subtitle_prefix=''):
    """Painel 2x2: (heatmap 2D, superfície 3D) para dois modelos, escala compartilhada.

    vmin/vmax compartilhados (vmax no percentil 95) deixam as duas superfícies na
    mesma escala de cor/altura, para comparar Adam e SAM de forma justa.
    """
    vmin = min(grid_a.min(), grid_b.min())
    vmax = max(np.percentile(grid_a, 95), np.percentile(grid_b, 95))
    pre = f'{subtitle_prefix} — ' if subtitle_prefix else ''

    fig, axes = plt.subplots(2, 2, figsize=(13, 10))
    for row, (grid, nome) in enumerate([(grid_a, name_a), (grid_b, name_b)]):
        ax_h = axes[row, 0]
        im = ax_h.contourf(alphas, betas, grid,
                           levels=30, cmap='RdYlBu_r', vmin=vmin, vmax=vmax)
        ax_h.contour(alphas, betas, grid,
                     levels=15, colors='black', linewidths=0.3, alpha=0.3)
        ax_h.scatter([0], [0], color='white', s=80, zorder=5,
                     edgecolors='black', linewidths=1.5)
        ax_h.set_xlabel('alpha (direction 1)', fontsize=10)
        ax_h.set_ylabel('beta (direction 2)', fontsize=10)
        ax_h.set_title(f'{pre}{nome}\n2D heatmap', fontsize=11)
        plt.colorbar(im, ax=ax_h, label='Loss')

        ax_3 = fig.add_subplot(2, 2, row * 2 + 2, projection='3d')
        A, B = np.meshgrid(alphas, betas)
        Z = np.clip(grid, vmin, vmax)
        ax_3.plot_surface(A, B, Z, cmap='RdYlBu_r', linewidth=0, antialiased=True, alpha=0.9)
        ax_3.set_zlim(vmin, vmax)
        ax_3.set_xlabel('alpha', fontsize=9); ax_3.set_ylabel('beta', fontsize=9)
        ax_3.set_zlabel('Loss', fontsize=9)
        ax_3.set_title(f'{pre}{nome}\n3D surface', fontsize=11)
        ax_3.view_init(elev=30, azim=-60)

    fig.suptitle(suptitle)
    fig.tight_layout(); plt.show()
