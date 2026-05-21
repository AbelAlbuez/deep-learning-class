# Proyecto Final — Brain Tumor Classification

Dataset: https://www.kaggle.com/datasets/sartajbhuvaji/brain-tumor-classification-mri

## 1. Crear entorno virtual

```
python -m venv venv
```

## 2. Activar entorno virtual

**Windows:**
```
venv\Scripts\activate
```

**Linux / Mac:**
```
source venv/bin/activate
```

## 3. Instalar dependencias

```
python -m pip install -r requirements.txt
```

## 4. Ejecutar EDA (local)

```
python src/data/eda.py --data_dir ruta/al/dataset
```

## 5. Entrenar modelos

```
python src/models/train.py --model 1 --data_dir ruta/al/dataset
python src/models/train.py --model 2 --data_dir ruta/al/dataset
python src/models/train.py --model 3 --data_dir ruta/al/dataset
```

## 6. Evaluar modelos

```
python src/models/evaluate.py --data_dir ruta/al/dataset
```

## 7. EDA en Google Colab

Abre `notebooks/eda_colab.ipynb` directamente en Colab.
El notebook descarga el dataset desde Kaggle automáticamente usando
las credenciales `kaggle.json`.
