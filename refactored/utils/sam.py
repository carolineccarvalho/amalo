"""SAM — Sharpness-Aware Minimization (Foret et al., 2021).

Em vez de minimizar L(θ), o SAM minimiza a pior perda numa vizinhança de raio ρ:
min_θ max_{||ε|| <= ρ} L(θ + ε). Isso empurra o treino para mínimos mais planos,
associados a melhor generalização. Cada passo de otimização tem duas fases:

  first_step  — sobe na direção do gradiente até ε* = ρ · g/||g|| (o ponto de
                maior perda na bola ρ) e guarda o deslocamento e_w;
  second_step — desfaz e_w (volta a θ) e deixa o otimizador base dar o passo
                usando o gradiente calculado no ponto perturbado θ + ε*.

Por isso o laço de treino chama model→backward duas vezes por batch.
"""

import torch


class SAM(torch.optim.Optimizer):
    def __init__(self, params, base_optimizer, rho=0.1, **kwargs):
        assert rho >= 0.0
        defaults = dict(rho=rho, **kwargs)
        super(SAM, self).__init__(params, defaults)
        self.base_optimizer = base_optimizer(self.param_groups, **kwargs)
        self.param_groups = self.base_optimizer.param_groups

    @torch.no_grad()
    def first_step(self, zero_grad=False):
        grad_norm = self._grad_norm()
        for group in self.param_groups:
            scale = group['rho'] / (grad_norm + 1e-12)
            for p in group['params']:
                if p.grad is None: continue
                e_w = p.grad * scale
                p.add_(e_w)
                self.state[p]['e_w'] = e_w
        if zero_grad: self.zero_grad()

    @torch.no_grad()
    def second_step(self, zero_grad=False):
        for group in self.param_groups:
            for p in group['params']:
                if p.grad is None: continue
                p.sub_(self.state[p]['e_w'])   # volta de θ+ε* para θ
        self.base_optimizer.step()
        if zero_grad: self.zero_grad()

    def _grad_norm(self):
        return torch.norm(torch.stack([
            p.grad.norm(p=2)
            for group in self.param_groups for p in group['params']
            if p.grad is not None
        ]), p=2)
