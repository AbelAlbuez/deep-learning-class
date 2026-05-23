# Clasificación de Tumores Cerebrales en MRI — CNN desde cero

> Proyecto Final · Aprendizaje Profundo 2026
> Maestría en Analítica para Inteligencia de Negocios (MAIN)
> Pontificia Universidad Javeriana, Bogotá D.C.
> Profesor: Julio Omar Palacio Niño, M.Sc.

## Integrantes

| Nombre | Responsabilidad |
|---|---|
| Abel Albuez Sánchez | EDA · Modelo 1 (CNN Baseline) · Coordinación |
| Daniel Rios | Modelo 2 (CNN Profunda) |
| Juan Torres | Modelo 2 (CNN Profunda) |
| Javier Esquivel | Modelo 3 (CNN Avanzada) · Tarea Investigación |

## Dataset

- **Nombre:** Brain Tumor Classification MRI
- **Fuente:** <https://www.kaggle.com/datasets/sartajbhuvaji/brain-tumor-classification-mri>
- **Clases:** `glioma_tumor` · `meningioma_tumor` · `no_tumor` · `pituitary_tumor`
- **Total imágenes:** 3 264 (2 870 *train* / 394 *test*)
- **Distribución por clase:**

| Clase | Train | Test | Total |
|---|---:|---:|---:|
| glioma_tumor | 826 | 100 | 926 |
| meningioma_tumor | 822 | 115 | 937 |
| no_tumor | 395 | 105 | 500 |
| pituitary_tumor | 827 | 74 | 901 |
| **Total** | **2 870** | **394** | **3 264** |

- **Ubicación local:** `proyecto-final/datasets/` (no versionada).
- **Restricción metodológica:** entrenamiento **desde cero**, sin *transfer learning*.

## Estructura del proyecto

```
proyecto-final/
├── README.md                       # Este archivo
├── requirements.txt                # Dependencias Python
├── reporte_eda.html                # Reporte EDA generado (HTML)
├── datasets/                       # Dataset Kaggle (no versionado)
│   ├── Training/{glioma,meningioma,no_tumor,pituitary}_tumor/
│   └── Testing/{glioma,meningioma,no_tumor,pituitary}_tumor/
├── notebooks/
│   └── eda_colab.ipynb             # EDA reproducible en Google Colab
├── src/
│   ├── data/
│   │   ├── dataset.py              # Dataset PyTorch + transforms
│   │   └── eda.py                  # Script local del EDA
│   ├── models/
│   │   ├── train.py                # Entrenamiento (--model 1|2|3)
│   │   └── evaluate.py             # Evaluación sobre Testing (holdout)
│   └── utils/
│       ├── metrics.py              # Accuracy, Precision, Recall, F1
│       └── plots.py                # Curvas y matrices de confusión
├── outputs/
│   ├── figures/                    # Figuras del EDA y de resultados (PNG)
│   └── results/                    # Checkpoints + metrics.csv
└── latex/
    ├── reporte/                    # Reporte técnico (IEEEtran, 2 col.)
    │   ├── main.tex
    │   ├── main.pdf
    │   ├── IEEEtran.cls
    │   └── secciones/
    │       ├── I_objetivo.tex
    │       ├── 02_eda.tex
    │       ├── III_metodologia.tex
    │       ├── IV_entrenamiento.tex
    │       ├── V_resultados.tex
    │       └── VI_conclusiones.tex
    └── articulo/
        └── main.tex                # Artículo académico (en desarrollo)
```

## Instalación y configuración

### 1. Clonar el repositorio

```bash
git clone https://github.com/AbelAlbuez/deep-learning-class.git
cd deep-learning-class/proyecto-final
```

### 2. Crear y activar entorno virtual

```bash
python -m venv venv
# macOS / Linux
source venv/bin/activate
# Windows
venv\Scripts\activate
```

### 3. Instalar dependencias

```bash
pip install -r requirements.txt
```

Dependencias principales: `torch`, `torchvision`, `numpy`, `pandas`,
`matplotlib`, `seaborn`, `scikit-learn`, `Pillow`, `tqdm`, `kaggle`.

### 4. Descargar el dataset

**Opción A — descarga manual (recomendada):**
descargar desde Kaggle y descomprimir en `proyecto-final/datasets/`
respetando la estructura `Training/<clase>/` y `Testing/<clase>/`.

**Opción B — vía API de Kaggle:**

```bash
# Configurar credenciales (NO commitear kaggle.json al repo)
mkdir -p ~/.kaggle && cp /ruta/a/kaggle.json ~/.kaggle/
chmod 600 ~/.kaggle/kaggle.json

# Descargar y descomprimir
kaggle datasets download -d sartajbhuvaji/brain-tumor-classification-mri \
  -p datasets/ --unzip
```

> ⚠️ **Seguridad:** `kaggle.json` contiene credenciales API personales.
> Mantenerlo siempre en `~/.kaggle/` y **nunca** versionarlo.

## Uso

### Ejecutar el EDA local

```bash
python src/data/eda.py --data_dir datasets
```

Genera figuras PNG en `outputs/figures/` y el reporte HTML
`reporte_eda.html`.

### EDA en Google Colab

Abrir `notebooks/eda_colab.ipynb` en Colab. El notebook descarga
automáticamente el dataset desde Kaggle usando `kaggle.json` cargado
como secreto del entorno Colab.

### Entrenar un modelo

```bash
# Modelo 1 — CNN Baseline (Abel)
python src/models/train.py --model 1 --data_dir datasets/Training

# Modelo 2 — CNN Profunda con BatchNorm + Dropout (Daniel, Juan)
python src/models/train.py --model 2 --data_dir datasets/Training

# Modelo 3 — CNN Avanzada (Javier)
python src/models/train.py --model 3 --data_dir datasets/Training
```

Cada entrenamiento:
- Aplica split interno **80/20** sobre `Training` para validación.
- Guarda *checkpoints* en `outputs/results/`.
- Registra métricas por época en `outputs/results/metrics.csv`.

### Evaluar sobre el holdout

```bash
python src/models/evaluate.py --model N --data_dir datasets/Testing
```

Reporta `Accuracy`, `Precision (macro)`, `Recall (macro)`,
`F1-Score (macro)` y matriz de confusión sobre el conjunto `Testing`.

## Pipeline de preprocesamiento

| # | Paso | Justificación |
|---|---|---|
| 1 | Resize 224×224 px | Resoluciones heterogéneas (de 59×80 a 1024×768). |
| 2 | Conversión a RGB | Algunas imágenes están en escala de grises. |
| 3 | Z-score (μ ImageNet) | Estabiliza el gradiente y mejora la convergencia. |
| 4 | Data augmentation | Rotación ±15°, *flip* horizontal y variación de brillo. |
| 5 | `class_weight` en pérdida | Compensa el desbalance de `no_tumor`. |
| 6 | Split 80/20 sobre `Training` | Valida sin contaminar el *holdout* oficial. |

## Métricas de evaluación

- `Accuracy`
- `Precision (macro)`
- `Recall (macro)`
- **`F1-Score (macro)` ← métrica principal** (robusta al desbalance)
- Matriz de confusión por modelo

## Reporte técnico (LaTeX)

El reporte se compila con `pdflatex` (TeX Live ≥ 2024). Estilo
IEEEtran *conference* (2 columnas), estructura modular en
`secciones/`.

```bash
cd latex/reporte
pdflatex -interaction=nonstopmode main.tex
pdflatex -interaction=nonstopmode main.tex   # 2.ª pasada para TOC/refs
```

Salida: `latex/reporte/main.pdf`. Las figuras se cargan desde
`../../outputs/figures/`, por lo que el PDF se actualiza
automáticamente cuando se regenera el EDA.

## Estado actual

- [x] EDA completo (script local + notebook Colab + reporte HTML)
- [x] Figuras EDA generadas
- [x] Reporte técnico LaTeX con secciones del EDA redactadas
- [ ] Modelo 1 — CNN Baseline (en desarrollo)
- [ ] Modelo 2 — CNN Profunda (en desarrollo)
- [ ] Modelo 3 — CNN Avanzada (en desarrollo)
- [ ] Tabla comparativa de resultados
- [ ] Conclusiones finales

## Convenciones de trabajo

- **Branch principal:** `main`.
- **Commits:** mensajes en español, formato
  `tipo(scope): descripción` (p. ej.
  `feat(modelo1): añadir CNN baseline`).
- **No versionar:** `datasets/`, `venv/`, `__pycache__/`,
  `kaggle.json`, artefactos LaTeX (`.aux`, `.log`, `.out`).
- **Antes de hacer push:** ejecutar el entrenamiento localmente
  y verificar que `outputs/results/metrics.csv` se genera sin
  errores.

