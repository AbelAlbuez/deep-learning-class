# deep-learning-class
## Taller 1 - Twitter User Gender Classification

Pontificia Universidad Javeriana  
Materia: Aprendizaje Profundo  
Profesor: Ing. Julio Omar Palacio Nino, M.Sc.  
Autores: Abel Albuez, Daniel Rios, Juan Torres, Javier Esquivel

---

## Descripcion

Clasificacion binaria de genero de usuarios de Twitter (male / female)
empleando redes neuronales feed-forward (MLP) en PyTorch.
Se descartan las clases `brand` y `unknown` por ser ambiguas.

---

## Estructura del proyecto

```
deep-learning-class/
├── README.md
├── requirements.txt
├── .gitignore
├── gender-classifier-DFE-791531.csv   <- dataset (no incluido en repo)
├── eda.py                             <- Punto 2: analisis exploratorio
├── preprocessing.py                   <- Puntos 3 y 4: limpieza y encoding
├── models.py                          <- Punto 6: arquitecturas MLP
├── train.py                           <- Puntos 5, 6 y 7: entrenamiento y evaluacion
├── dataset_clean.csv                  <- output de preprocessing.py (no incluido)
├── informe_taller1.docx               <- informe completo del taller
├── plots/                             <- graficas generadas automaticamente
└── models/                            <- pesos guardados (.pt)
```

---

## Requisitos

- Python 3.12
- Dataset: https://www.kaggle.com/crowdflower/twitter-user-gender-classification

---

## Configuracion del entorno virtual

### 1. Crear el entorno virtual

```bash
python3 -m venv venv
```

### 2. Activar el entorno virtual

**macOS / Linux:**
```bash
source venv/bin/activate
```

**Windows:**
```bash
venv\Scripts\activate
```

### 3. Instalar dependencias

```bash
pip install -r requirements.txt
```

### 4. Verificar instalacion

```bash
python3 -c "import torch; print('PyTorch:', torch.__version__)"
python3 -c "import sklearn; print('Scikit-learn:', sklearn.__version__)"
```

---

## Ejecucion

Orden de ejecucion recomendado:

```bash
# 1. Analisis exploratorio
python3 eda.py

# 2. Preprocesamiento
python3 preprocessing.py

# 3. Verificar arquitecturas
python3 models.py

# 4. Entrenamiento y evaluacion
python3 train.py
```

---

## Resultados

| Modelo            | Accuracy | Precision | Recall | F1    |
|-------------------|----------|-----------|--------|-------|
| Perceptron        | 0.632    | 0.634     | 0.668  | 0.651 |
| MLP 1 capa oculta | 0.630    | 0.630     | 0.685  | 0.656 |
| MLP 2 capas ocultas | 0.638  | 0.638     | 0.690  | 0.663 |

---

## Desactivar el entorno virtual

```bash
deactivate
```