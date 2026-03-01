# deep-learning-class
## Taller 1 - Twitter User Gender Classification

## Autores

| Nombre          |
|-----------------|
| Abel Albuez     |
| Daniel Rios     |
| Juan Torres     |
| Javier Esquivel |

Pontificia Universidad Javeriana  
Materia: Aprendizaje Profundo  
Profesor: Ing. Julio Omar Palacio Nino, M.Sc.

---

## Descripcion

Analisis de clasificacion de genero de usuarios de Twitter empleando redes
neuronales feed-forward (FFNN - MLP). Se trabaja con clasificacion binaria
(male / female) descartando las clases `brand` y `unknown`.

---

## Estructura del proyecto

```
deep-learning-class/
├── README.md
├── requirements.txt
├── gender-classifier-DFE-791531.csv   <- dataset descargado de Kaggle
├── eda.py                             <- Punto 2: analisis exploratorio
├── preprocessing.py                   <- Punto 3 y 4: limpieza y encoding
├── train.py                           <- Puntos 5, 6 y 7: modelos y evaluacion
├── plots/                             <- graficas generadas automaticamente
└── models/                            <- modelos entrenados guardados
```

---

## Requisitos

- Python 3.10 o superior
- Dataset: https://www.kaggle.com/crowdflower/twitter-user-gender-classification

---

## Configuracion del entorno virtual

### 1. Crear el entorno virtual

```bash
python -m venv venv
```

### 2. Activar el entorno virtual

**Windows:**
```bash
venv\Scripts\activate
```

**macOS / Linux:**
```bash
source venv/bin/activate
```

### 3. Instalar dependencias

```bash
pip install -r requirements.txt
```

### 4. Verificar instalacion

```bash
python -c "import torch; print('PyTorch:', torch.__version__)"
python -c "import sklearn; print('Scikit-learn:', sklearn.__version__)"
```

---

## Ejecucion

Asegurate de tener el entorno virtual activado antes de correr cualquier script.

### Punto 2 - Analisis exploratorio (EDA)

```bash
python eda.py
```

Genera graficas en la carpeta `plots/`.

### Punto 3 y 4 - Preprocesamiento

```bash
python preprocessing.py
```

Genera el dataset limpio listo para entrenar.

### Puntos 5, 6 y 7 - Entrenamiento y evaluacion

```bash
python train.py
```

Entrena los tres modelos y genera las matrices de confusion y metricas.

---

## Desactivar el entorno virtual

Cuando termines de trabajar:

```bash
deactivate
```

---

## Notas

- El archivo CSV debe estar en la raiz del proyecto con el nombre exacto:
  `gender-classifier-DFE-791531.csv`
- Las carpetas `plots/` y `models/` se crean automaticamente al ejecutar los scripts.
- Todos los scripts fueron desarrollados y probados con Python 3.10.
- Los modelos MLP estan implementados en PyTorch.

