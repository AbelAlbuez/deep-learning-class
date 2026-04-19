# Clasificador de Señales de Tráfico con CNN

![Python](https://img.shields.io/badge/python-3.10-blue.svg)
![TensorFlow](https://img.shields.io/badge/tensorflow-2.x-orange.svg)
![Licencia](https://img.shields.io/badge/licencia-academica-lightgrey.svg)

> Redes neuronales convolucionales entrenadas desde cero para clasificar señales de tráfico chinas en 10 categorías, con un barrido completo de hiperparámetros.

---

## Información Académica

- **Universidad**: Pontificia Universidad Javeriana
- **Programa**: Maestría en Analítica para Inteligencia de Negocios
- **Curso**: Aprendizaje Profundo (Deep Learning)
- **Profesor**: Julio Omar Palacio Niño, M.Sc.
- **Autores**:
  - Abel Albuez
  - Daniel Rios
  - Juan Torres
  - Javier Esquivel

---

## Descripción General

Este proyecto evalúa cómo dos arquitecturas CNN de profundidad distinta se comportan sobre un dataset real y fuertemente **desbalanceado** de señales de tráfico. Todos los modelos se entrenan **desde cero** — sin transfer learning ni pesos preentrenados — para aislar el efecto de la arquitectura y los hiperparámetros sobre la generalización.

El pipeline:

1. Explora y caracteriza el dataset (distribución de clases, estadísticas de imagen).
2. Construye dos CNN con profundidades crecientes.
3. Ejecuta una búsqueda en grilla sobre `epochs × batch_size × modelo` (24 combinaciones) con soporte automático de **reanudación** para continuar ejecuciones interrumpidas.
4. Registra cada experimento en un CSV y genera gráficos por corrida (curvas de pérdida/accuracy + matrices de confusión).
5. Resume las mejores configuraciones según accuracy de validación, accuracy de test y F1 ponderado.
6. Incluye un modelo bonus con BatchNormalization, Dropout y data augmentation.

---

## Dataset

- **Fuente**: Kaggle — Chinese Traffic Sign Dataset (señales de tráfico chinas).
- **Clases (10)**: `GuideSign`, `M1`, `M4`, `M5`, `M6`, `M7`, `P1`, `P10_50`, `P12`, `W1`.
- **Resolución**: todas las imágenes 224×224 RGB (JPEG). Se redimensionan a **32×32** antes de alimentar la red.
- **Distribución de clases**: fuertemente desbalanceada. `M4` concentra aproximadamente el 50 % de las muestras de entrenamiento, mientras que `P10_50`, `P12`, `M5` y `M6` apenas superan el 4 % cada una. Este desbalance es la restricción más importante que los modelos deben superar.

```
GuideSign  1171
M1          247
M4         3206   ← clase dominante
M5          213
M6          134   ← clase más escasa en train
M7          469
P1          249
P10_50       95
P12          95
W1          145
```

---

## Arquitecturas

Ambos modelos usan Adam (`lr=1e-3`), `categorical crossentropy` y salida softmax sobre 10 clases. La entrada es `(32, 32, 3)` con píxeles normalizados a `[0, 1]`.

### Modelo 1 — CNN Simple

```
Input(32, 32, 3)
Conv2D(32, 5x5, ReLU, stride=1, padding='same')
MaxPooling2D(5x5)
Flatten
Dense(100, ReLU)
Dense(10, Softmax)
```

### Modelo 2 — CNN Profunda

```
Input(32, 32, 3)
Conv2D(48, 3x3, ReLU, padding='same')
MaxPooling2D(2x2)
Conv2D(96, 3x3, ReLU, padding='same')
MaxPooling2D(2x2)
Flatten
Dense(100, ReLU)
Dense(100, ReLU)
Dense(10, Softmax)
```

### Bonus — Modelo 2 con regularización

El Modelo 2 aumentado con `BatchNormalization` después de cada convolución, `Dropout(0.3)` entre bloques y capas densas, y data augmentation en tiempo de entrenamiento (rotación, zoom, brillo y desplazamiento). Se entrena con callbacks `EarlyStopping` y `ReduceLROnPlateau`.

---

## Grilla de Experimentos

Definida al inicio de `main.py`:

```python
EXPERIMENT_GRID = {
    "epochs":      [10, 20, 30],
    "batch_sizes": [4, 16, 32, 64],
    "models":      ["model1", "model2"],
}
```

- **24 combinaciones en total** (3 × 4 × 2).
- La iteración alterna `model1` ↔ `model2` para cada par `(epochs, batch_size)`, de modo que ambas arquitecturas se comparan consecutivamente bajo condiciones idénticas.
- Cada experimento entrena, evalúa en el conjunto de test y añade una fila a `outputs/results.csv`.
- **Soporte de reanudación**: al arrancar, el script lee `results.csv` y omite cualquier tripleta `(model, epochs, batch_size)` ya registrada. Es seguro interrumpir y volver a ejecutar en cualquier momento.

---

## Estructura del Proyecto

```
traffic-sign-cnn-classifier/
├── main.py                     # Driver de experimentos end-to-end
├── taller_2.ipynb              # Versión notebook (exploración + walkthrough)
├── requirements.txt
├── README.md
├── datasets/
│   ├── train_dataset/train/<clase>/*.jpg
│   └── test_dataset/test/<clase>/*.jpg
└── outputs/                    # Se crea automáticamente en la primera corrida
    ├── exploration/            # Distribución de clases y muestras por clase
    ├── models/                 # {modelo}_e{epochs}_bs{batch_size}.keras
    │                           # + model_bonus.keras
    ├── plots/                  # {tag}_curves.png y {tag}_cm.png
    ├── results.csv             # Una fila por experimento
    ├── summary.txt             # Mejores corridas + tabla ordenada
    └── run.log                 # Salida completa (vía tee)
```

---

## Instalación y Ejecución

Requiere Python 3.10 (arm64 en Apple Silicon). Desde el directorio del proyecto:

```bash
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
python main.py
```

Para guardar un log de la ejecución:

```bash
python main.py 2>&1 | tee outputs/run.log
```

Volver a ejecutar el script tras una interrupción reanuda desde la última fila registrada en `outputs/results.csv`.

---

## Resultados

El mejor modelo fue **model2_e20_bs64** con test_acc=0.9722 y f1=0.9703.

| Config | test_acc | f1_score | val_acc | Tiempo |
|--------|----------|----------|---------|--------|
| model2_e20_bs64 | 0.9722 | 0.9703 | 0.9550 | 44.5s |
| model1_e30_bs4  | 0.9691 | 0.9685 | 0.9559 | 78.3s |
| model2_e30_bs32 | 0.9691 | 0.9679 | 0.9584 | 75.2s |
| model1_e10_bs16 | 0.9660 | 0.9641 | 0.9542 | 19.7s |
| model1_e20_bs4  | 0.9660 | 0.9655 | 0.9567 | 52.4s |

Las métricas completas por experimento están en [`outputs/results.csv`](outputs/results.csv) y el reporte ordenado en [`outputs/summary.txt`](outputs/summary.txt). Los diagnósticos por corrida (curvas de pérdida/accuracy y matriz de confusión) se guardan en `outputs/plots/{modelo}_e{epochs}_bs{bs}_curves.png` y `..._cm.png`.

---

## Análisis por clase

Si bien el accuracy global del mejor modelo alcanza 0.9722, el análisis por clase revela una debilidad crítica en **M6** (f1=0.667), clase que representa solo 8 imágenes en test y cuya apariencia visual es similar a otras señales azules de maniobra. En contraste, clases con características visuales distintivas como **W1** (triángulo amarillo) y **P10_50** (círculo rojo) alcanzan f1=1.0 en ambos modelos.

| Clase | Precision | Recall | F1 | Support |
|-------|-----------|--------|----|---------|
| GuideSign | 0.9839 | 0.9839 | 0.9839 | 62 |
| M1 | 0.8571 | 0.8571 | 0.8571 | 14 |
| M4 | 0.9767 | 0.9941 | 0.9853 | 169 |
| M5 | 1.0000 | 1.0000 | 1.0000 | 12 |
| M6 | 1.0000 | 0.5000 | 0.6667 | 8 |
| M7 | 0.9600 | 0.9600 | 0.9600 | 25 |
| P1 | 0.9333 | 1.0000 | 0.9655 | 14 |
| P10_50 | 1.0000 | 1.0000 | 1.0000 | 6 |
| P12 | 1.0000 | 1.0000 | 1.0000 | 6 |
| W1 | 1.0000 | 1.0000 | 1.0000 | 8 |

> Nota: las métricas globales están infladas por las clases mayoritarias (M4, GuideSign). El desbalance severo del dataset es el principal factor que limita el rendimiento en clases minoritarias.

---

## Hallazgos Clave

- **El Modelo 2 aprende las clases minoritarias más rápido.** Con dos bloques convolucionales y features jerárquicos, alcanza accuracy aceptable sobre clases poco representadas (`P10_50`, `P12`, `M6`) en menos épocas que el Modelo 1, cuyo pooling 5×5 descarta información espacial demasiado pronto.
- **`batch_size = 16` es el punto más estable** en ambas arquitecturas: produce curvas de validación suaves y rendimiento consistente en test, sin el ruido de gradiente de `bs=4` ni la actualización lenta de `bs=64`.
- **`M6` es la clase más difícil.** Combina pocas muestras (~2 % del train) con similitud visual a otras señales de advertencia (`W1`), por lo que ambos modelos fallan más en ella. Un aumento del recall en `M6` es el mejor indicador de que una técnica de regularización o augmentación está funcionando.
- El fuerte desbalance (`M4` ≈ 50 % del train) infla la accuracy bruta; **el F1 ponderado es la métrica más confiable** para comparar corridas.

---

## Notas

- No se utiliza transfer learning ni modelos preentrenados en ninguna parte del proyecto. Todos los pesos se aprenden desde cero con el dataset provisto.
- Reproducibilidad: todas las semillas aleatorias están fijadas a `42` (`numpy`, `tensorflow`, `random`). El orden de carga de datos es determinista dada la semilla.
