import torch
import torch.nn as nn


class Perceptron(nn.Module):
    def __init__(self, n_features: int):
        super().__init__()
        self.linear = nn.Linear(n_features, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return (self.linear(x) >= 0.0).float()


class MLP1Hidden(nn.Module):
    def __init__(self, n_features: int, dropout_p: float = 0.3):
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(n_features, n_features),
            nn.ReLU(),
            nn.Dropout(dropout_p),
            nn.Linear(n_features, 1),
            nn.Sigmoid()
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.network(x)


class MLP2Hidden(nn.Module):
    def __init__(self, n_features: int, dropout_p: float = 0.3):
        super().__init__()
        n_hidden = n_features // 2
        self.network = nn.Sequential(
            nn.Linear(n_features, n_hidden),
            nn.ReLU(),
            nn.Dropout(dropout_p),
            nn.Linear(n_hidden, n_hidden),
            nn.ReLU(),
            nn.Dropout(dropout_p),
            nn.Linear(n_hidden, 1),
            nn.Sigmoid()
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.network(x)


def get_model(name: str, n_features: int) -> nn.Module:
    models = {
        "perceptron": Perceptron,
        "mlp1":       MLP1Hidden,
        "mlp2":       MLP2Hidden,
    }
    if name not in models:
        raise ValueError(f"Modelo no reconocido: {name}. Opciones: {list(models.keys())}")
    return models[name](n_features)


def print_summary(model: nn.Module, name: str):
    total     = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"\n{name}")
    print(f"  {model}")
    print(f"  Parametros totales    : {total:,}")
    print(f"  Parametros entrenables: {trainable:,}")


if __name__ == "__main__":
    N = 27

    perceptron = get_model("perceptron", N)
    mlp1       = get_model("mlp1",       N)
    mlp2       = get_model("mlp2",       N)

    print_summary(perceptron, "Perceptron unicapa")
    print_summary(mlp1,       "MLP 1 capa oculta")
    print_summary(mlp2,       "MLP 2 capas ocultas")

    x = torch.randn(4, N)
    print("\nVerificacion de salidas (batch=4):")
    print(f"  Perceptron : {perceptron(x).squeeze().tolist()}")
    print(f"  MLP1       : {mlp1(x).squeeze().tolist()}")
    print(f"  MLP2       : {mlp2(x).squeeze().tolist()}")
