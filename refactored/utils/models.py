"""Arquiteturas: MLP (dados tabulares/achatados) e CNN (imagens)."""

import torch.nn as nn


class MLP(nn.Module):
    """Perceptron multicamada com ReLU nas ocultas e saída linear (logits).

    A primeira camada é um Flatten, então a MLP também aceita imagens (achatadas).
    A saída fica sem ativação: usamos os logits direto na perda
    (BCEWithLogitsLoss / CrossEntropyLoss).
    """
    def __init__(self, input_dim, hidden_dims, output_dim):
        super().__init__()
        layers = [nn.Flatten()]
        dims = [input_dim] + hidden_dims + [output_dim]
        for i in range(len(dims) - 1):
            layers.append(nn.Linear(dims[i], dims[i+1]))
            if i < len(dims) - 2:        # sem ReLU após a camada de saída
                layers.append(nn.ReLU())
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)


class CNN(nn.Module):
    """CNN para imagens: 2 blocos Conv+ReLU+MaxPool e uma camada FC oculta."""
    def __init__(self, in_channels, img_size, num_classes):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(in_channels, 32, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
        )
        # Dois MaxPool(2) reduzem cada dimensão espacial por 4; 64 canais na saída.
        feat_size = (img_size // 4) ** 2 * 64
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(feat_size, 256), nn.ReLU(),
            nn.Linear(256, num_classes),
        )

    def forward(self, x):
        return self.classifier(self.features(x))
