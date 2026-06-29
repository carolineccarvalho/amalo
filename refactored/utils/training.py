"""Seeding e laço de treino com early stopping."""

import copy
import random

import numpy as np
import torch


def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def _contar_acertos(out, y):
    # A mesma rede serve tarefas binárias (1 logit) e multiclasse (C logits).
    # Detectamos qual pela forma da saída para não precisar de um flag de tarefa:
    #   - multiclasse: argmax sobre as classes;
    #   - binária: logit > 0 (limiar 0 equivale a probabilidade 0.5), comparado
    #     ao alvo float de forma [N,1] (sem squeeze, para o broadcast bater).
    if out.dim() > 1 and out.shape[1] > 1:
        return (out.argmax(1) == y).sum().item()
    return ((out > 0.0).float() == y).sum().item()


def train_model(model, train_loader, val_loader, optimizer, criterion, epochs,
                is_sam=False, patience=30, restore_best=True):
    """Treina `model` e devolve o histórico de loss/acurácia por época.

    `criterion` define a perda (BCEWithLogitsLoss para binário, CrossEntropyLoss
    para multiclasse); a acurácia é inferida pela forma da saída.

    SAM exige dois passos por batch: o primeiro sobe o gradiente até o ponto de
    pior perda na vizinhança ρ (first_step) e o segundo aplica o otimizador base
    a partir desse ponto perturbado (second_step) — daí o `is_sam`.

    Com `restore_best`, ao final recarregamos os pesos da época de menor val_loss
    (early stopping com paciência `patience`), evitando reportar um modelo já
    sobreajustado.
    """
    history = {'train_loss': [], 'val_loss': [], 'train_acc': [], 'val_acc': []}
    best_state = copy.deepcopy(model.state_dict())
    best_val_loss = float('inf')
    no_improve = 0

    for epoch in range(epochs):
        model.train()
        t_loss = t_correct = t_total = 0
        for X, y in train_loader:
            if is_sam:
                out = model(X); loss = criterion(out, y); loss.backward()
                optimizer.first_step(zero_grad=True)
                out2 = model(X); criterion(out2, y).backward()
                optimizer.second_step(zero_grad=True)
                with torch.no_grad():
                    out_log = model(X); loss_log = criterion(out_log, y)
            else:
                optimizer.zero_grad()
                out_log = model(X); loss_log = criterion(out_log, y)
                loss_log.backward(); optimizer.step()
            t_loss += loss_log.item()
            t_correct += _contar_acertos(out_log, y)
            t_total += y.size(0)

        model.eval()
        v_loss = v_correct = v_total = 0
        with torch.no_grad():
            for X, y in val_loader:
                out = model(X); loss = criterion(out, y)
                v_loss += loss.item()
                v_correct += _contar_acertos(out, y)
                v_total += y.size(0)

        avg_v = v_loss / len(val_loader)
        history['train_loss'].append(t_loss / len(train_loader))
        history['val_loss'].append(avg_v)
        history['train_acc'].append(t_correct / t_total)
        history['val_acc'].append(v_correct / v_total)

        if avg_v < best_val_loss:
            best_val_loss = avg_v
            best_state = copy.deepcopy(model.state_dict())
            no_improve = 0
        else:
            no_improve += 1
            if patience is not None and no_improve >= patience:
                break

    if restore_best:
        model.load_state_dict(best_state)
    history['epochs_trained'] = len(history['train_loss'])
    history['best_val_loss'] = best_val_loss
    return history
