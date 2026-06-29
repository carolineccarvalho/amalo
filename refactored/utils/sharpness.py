"""Sharpness do mínimo: maior autovalor da Hessiana (λ_max).

Instanciar a Hessiana de uma rede é inviável. Usamos o truque de Pearlmutter para
o produto Hessiana-vetor (Hv) via dupla diferenciação automática — sem montar H —
e iteramos o método da potência: v ← Hv/||Hv|| converge para o autovetor dominante,
e v·Hv para λ_max. Um λ_max alto indica um mínimo agudo.
"""

import torch


def calcular_autovalor_dominante_full(inputs, targets, model, criterion,
                                      num_iterations=20, seed=None):
    """Método da potência sobre a Hessiana. Retorna (histórico de λ por iteração, autovetor)."""
    model.eval()
    if seed is not None:
        torch.manual_seed(seed)
    num_params = sum(p.numel() for p in model.parameters())
    v = torch.randn(num_params, device=inputs.device)
    v = v / torch.norm(v)
    history_eigenvalues = []
    for _ in range(num_iterations):
        model.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, targets)
        # grad com create_graph=True permite derivar de novo para obter Hv.
        grads = torch.autograd.grad(loss, model.parameters(), create_graph=True)
        flat_grads = torch.cat([g.contiguous().view(-1) for g in grads])
        grad_v_prod = torch.dot(flat_grads, v)
        hvp = torch.autograd.grad(grad_v_prod, model.parameters())  # = H v
        flat_hvp = torch.cat([h.contiguous().view(-1) for h in hvp])
        eigenvalue = torch.dot(v, flat_hvp).item()                  # quociente de Rayleigh
        history_eigenvalues.append(eigenvalue)
        hvp_norm = torch.norm(flat_hvp)
        if hvp_norm.item() < 1e-12:
            break
        v = flat_hvp / hvp_norm
    return history_eigenvalues, v.detach()


def get_lambda_max(model, inputs, targets, criterion, num_iterations=20, seed=None):
    """λ_max (valor convergido do método da potência)."""
    history, _ = calcular_autovalor_dominante_full(
        inputs, targets, model, criterion,
        num_iterations=num_iterations, seed=seed)
    return history[-1]


def power_method_curve(model, inputs, targets, criterion, num_iterations, seed=None):
    """Histórico de λ por iteração (para inspecionar a convergência)."""
    history, _ = calcular_autovalor_dominante_full(
        inputs, targets, model, criterion,
        num_iterations=num_iterations, seed=seed)
    return history
