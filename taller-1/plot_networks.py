# =============================================================
# deep-learning-class - Visualización de redes neuronales
# Genera un gráfico PNG con las 3 arquitecturas: Perceptron, MLP1, MLP2
# =============================================================

import matplotlib.pyplot as plt
import numpy as np
import os

# Arquitecturas del proyecto (models.py)
N_FEATURES = 27
N_HIDDEN_MLP2 = N_FEATURES // 2  # 13

OUTPUT_DIR = "plots"
os.makedirs(OUTPUT_DIR, exist_ok=True)


def draw_layer(ax, x, y_positions, label="", color="steelblue", is_output=False):
    """Dibuja un conjunto de neuronas (círculos) en una columna."""
    for i, y in enumerate(y_positions):
        circle = plt.Circle((x, y), 0.12, color=color, ec="black", linewidth=0.8, zorder=2)
        ax.add_patch(circle)
    if label:
        ax.text(x, y_positions[-1] - 0.45, label, ha="center", fontsize=8, fontweight="bold")


def draw_connections(ax, x_left, y_left, x_right, y_right, alpha=0.4):
    """Dibuja líneas entre dos capas."""
    for yl in y_left:
        for yr in y_right:
            ax.plot([x_left + 0.12, x_right - 0.12], [yl, yr], "k-", alpha=alpha, lw=0.5, zorder=1)


def layout_neurons(n_neurons, y_min=-1.5, y_max=1.5):
    """Distribuye n neuronas en el eje Y."""
    if n_neurons == 1:
        return [0.0]
    return np.linspace(y_max, y_min, n_neurons)


def plot_architecture(ax, layers, title, layer_labels=None):
    """
    layers: lista de tamaños [input, hidden1, ..., output]
    layer_labels: opcional, nombres para cada capa
    """
    if layer_labels is None:
        layer_labels = [str(l) for l in layers]
    n_cols = len(layers)
    x_positions = np.linspace(0, 4, n_cols)
    colors = ["#2ecc71", "#3498db", "#9b59b6", "#e74c3c"]  # verde, azul, violeta, rojo

    for i, (n_neurons, x) in enumerate(zip(layers, x_positions)):
        y_positions = layout_neurons(n_neurons)
        is_output = i == n_cols - 1
        color = colors[min(i, len(colors) - 1)]
        if is_output:
            color = "#e74c3c"
        draw_layer(ax, x, y_positions, label=layer_labels[i], color=color, is_output=is_output)

    for i in range(n_cols - 1):
        y_left = layout_neurons(layers[i])
        y_right = layout_neurons(layers[i + 1])
        draw_connections(ax, x_positions[i], y_left, x_positions[i + 1], y_right)

    ax.set_xlim(-0.5, 4.5)
    ax.set_ylim(-2.2, 2.2)
    ax.set_aspect("equal")
    ax.axis("off")
    ax.set_title(title, fontsize=12, fontweight="bold")


def main():
    fig, axes = plt.subplots(1, 3, figsize=(14, 5))

    # Perceptron: 27 -> 1 (mostramos 5 nodos de entrada como representación)
    plot_architecture(
        axes[0],
        [5, 1],
        "Perceptron (1 capa)",
        layer_labels=["27 in", "1 out"],
    )
    axes[0].text(2, -1.9, "27 entradas → 1 salida (TLU)", ha="center", fontsize=9, style="italic")

    # MLP 1 oculta: 27 -> 27 -> 1 (representamos con menos nodos para claridad)
    plot_architecture(
        axes[1],
        [5, 5, 1],
        "MLP 1 capa oculta",
        layer_labels=["27 in", "27 H", "1 out"],
    )
    axes[1].text(2, -1.9, "ReLU + Dropout → Sigmoid", ha="center", fontsize=9, style="italic")

    # MLP 2 ocultas: 27 -> 13 -> 13 -> 1
    plot_architecture(
        axes[2],
        [5, 4, 4, 1],
        "MLP 2 capas ocultas",
        layer_labels=["27 in", "13 H1", "13 H2", "1 out"],
    )
    axes[2].text(2, -1.9, "ReLU + Dropout en cada capa", ha="center", fontsize=9, style="italic")

    plt.suptitle("Arquitecturas del clasificador de género (Twitter)", fontsize=14, fontweight="bold", y=1.02)
    plt.tight_layout()
    out_path = os.path.join(OUTPUT_DIR, "06_redes_neuronales.png")
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Gráfico guardado: {out_path}")


if __name__ == "__main__":
    main()
