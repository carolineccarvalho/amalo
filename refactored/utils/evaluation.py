"""Avaliação de modelos (loss + acurácia)."""

import torch


def _contar_acertos(out, y):
    # Binário (1 logit) vs multiclasse (C logits) detectado pela forma da saída.
    if out.dim() > 1 and out.shape[1] > 1:
        return (out.argmax(1) == y).sum().item()
    return ((out > 0.0).float() == y).sum().item()


def evaluate(model, loader, criterion):
    """Loss média e acurácia de `model` sobre `loader`."""
    model.eval()
    total_loss = correct = total = 0
    with torch.no_grad():
        for X, y in loader:
            out = model(X)
            total_loss += criterion(out, y).item()
            correct += _contar_acertos(out, y)
            total += y.size(0)
    return {'loss': total_loss / len(loader), 'accuracy': correct / total}


def resumir_resultados(nome, model, tr_loader, val_loader, te_loader, criterion):
    """Avalia treino/val/teste e o gap de generalização (acc treino − acc teste)."""
    tm = evaluate(model, tr_loader, criterion)
    vm = evaluate(model, val_loader, criterion)
    sm = evaluate(model, te_loader, criterion)
    gap = tm['accuracy'] - sm['accuracy']
    print(f'{nome}')
    print(f"  treino: loss={tm['loss']:.4f}  acc={tm['accuracy']:.3f}")
    print(f"  val:    loss={vm['loss']:.4f}  acc={vm['accuracy']:.3f}")
    print(f"  teste:  loss={sm['loss']:.4f}  acc={sm['accuracy']:.3f}")
    print(f'  gap (treino-teste acc): {gap:.3f}')
    return {'train': tm, 'val': vm, 'test': sm, 'gap': gap}
