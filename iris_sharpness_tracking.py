# %% ----------------------------------------------------------------------
# Iris: lambda_max ao longo do treinamento (Adam vs SAM + Adam)
#
# Mesma ideia ja usada no Titanic (train_with_sharpness_tracking +
# plot_sharpness_over_time, importadas de loss_geometria.py), mas adaptada
# para classificacao multiclasse (CrossEntropyLoss em vez de
# BCEWithLogitsLoss e accuracy via argmax em vez de threshold 0).
#
# Cole esta celula depois do treino Adam/SAM do Iris no notebook.
# ------------------------------------------------------------------------
import copy
from loss_geometria import calcular_autovalor_dominante_full, plot_sharpness_over_time


def train_multiclass_with_sharpness_tracking(model, train_loader, val_loader,
                                              optimizer, criterion,
                                              epochs=300, is_sam=False,
                                              patience=40, track_every=10,
                                              X_full=None, y_full=None,
                                              restore_best=True, lambda_seed=42):
    """
    Igual a train_model_multiclass do notebook, mas rastreando lambda_max
    a cada `track_every` epocas (mesmo padrao de train_with_sharpness_tracking,
    so que com CrossEntropyLoss e accuracy por argmax).
    """
    history = {
        'train_loss': [], 'val_loss': [],
        'train_acc': [], 'val_acc': [],
        'lambda_max_history': []
    }
    best_state = copy.deepcopy(model.state_dict())
    best_val_loss = float('inf')
    no_improve = 0

    for epoch in range(epochs):
        model.train()
        t_loss, t_correct, t_total = 0.0, 0, 0
        for X, y in train_loader:
            if is_sam:
                out = model(X); loss = criterion(out, y); loss.backward()
                optimizer.first_step(zero_grad=True)
                out2 = model(X); loss2 = criterion(out2, y); loss2.backward()
                optimizer.second_step(zero_grad=True)
                with torch.no_grad():
                    out_log = model(X); loss_log = criterion(out_log, y)
            else:
                optimizer.zero_grad()
                out_log = model(X); loss_log = criterion(out_log, y)
                loss_log.backward(); optimizer.step()
            t_loss += loss_log.item()
            t_correct += (out_log.argmax(1) == y).sum().item()
            t_total += y.size(0)

        model.eval()
        v_loss, v_correct, v_total = 0.0, 0, 0
        with torch.no_grad():
            for X, y in val_loader:
                out = model(X); loss = criterion(out, y)
                v_loss += loss.item()
                v_correct += (out.argmax(1) == y).sum().item()
                v_total += y.size(0)

        avg_v = v_loss / len(val_loader)
        history['train_loss'].append(t_loss / len(train_loader))
        history['val_loss'].append(avg_v)
        history['train_acc'].append(t_correct / t_total)
        history['val_acc'].append(v_correct / v_total)

        # Rastreia lambda_max periodicamente (mesma logica do Titanic)
        if X_full is not None and (epoch + 1) % track_every == 0:
            lam = calcular_autovalor_dominante_full(
                X_full, y_full, model, criterion, num_iterations=10, seed=lambda_seed
            )
            history['lambda_max_history'].append((epoch + 1, lam))
            print(f"  Epoca {epoch + 1}: lambda_max = {lam:.4f}")

        if avg_v < best_val_loss:
            best_val_loss = avg_v
            best_state = copy.deepcopy(model.state_dict())
            no_improve = 0
        else:
            no_improve += 1
            if patience and no_improve >= patience:
                print(f"  Early stopping na epoca {epoch + 1}.")
                break

    if restore_best:
        model.load_state_dict(best_state)
    history['epochs_trained'] = len(history['train_loss'])
    return history


# ---- Treino com tracking -----------------------------------------------
set_seed(42)
modelo_adam_iris_track = MLP(input_dim=input_dim_iris, hidden_dims=[16, 16], output_dim=3)
opt_adam_iris_track = optim.Adam(modelo_adam_iris_track.parameters(), lr=0.01, weight_decay=1e-4)

historico_adam_iris_track = train_multiclass_with_sharpness_tracking(
    model=modelo_adam_iris_track,
    train_loader=train_loader_iris,
    val_loader=val_loader_iris,
    optimizer=opt_adam_iris_track,
    criterion=nn.CrossEntropyLoss(),
    epochs=300,
    is_sam=False,
    patience=40,
    track_every=10,
    X_full=X_train_iris_t,
    y_full=y_train_iris_t,
)

set_seed(42)
modelo_sam_iris_track = MLP(input_dim=input_dim_iris, hidden_dims=[16, 16], output_dim=3)
opt_sam_iris_track = SAM(modelo_sam_iris_track.parameters(), base_optimizer=optim.Adam,
                          lr=0.01, weight_decay=1e-4, rho=0.05)

historico_sam_iris_track = train_multiclass_with_sharpness_tracking(
    model=modelo_sam_iris_track,
    train_loader=train_loader_iris,
    val_loader=val_loader_iris,
    optimizer=opt_sam_iris_track,
    criterion=nn.CrossEntropyLoss(),
    epochs=300,
    is_sam=True,
    patience=40,
    track_every=10,
    X_full=X_train_iris_t,
    y_full=y_train_iris_t,
)

plot_sharpness_over_time(
    {'Adam': historico_adam_iris_track, 'SAM + Adam (rho=0.05)': historico_sam_iris_track},
    title='Iris - Evolucao de lambda_max durante o treinamento'
)
