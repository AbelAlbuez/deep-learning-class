import os
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models, optimizers
from tensorflow.keras.applications.vgg16 import VGG16, preprocess_input
from sklearn.metrics import classification_report, confusion_matrix

IMG_SIZE = (224, 224)
BATCH_SIZE = 32
TRAIN_DIR = '../deep/deep-learning-class/proyecto-final/datasets/Training/'
TEST_DIR = '../deep/deep-learning-class/proyecto-final/datasets/Testing/'

train_ds = tf.keras.utils.image_dataset_from_directory(
    TRAIN_DIR,
    validation_split=0.2,
    subset="training",
    seed=123,
    image_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    label_mode='categorical'
)

val_ds = tf.keras.utils.image_dataset_from_directory(
    TRAIN_DIR,
    validation_split=0.2,
    subset="validation",
    seed=123,
    image_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    label_mode='categorical'
)

test_ds = tf.keras.utils.image_dataset_from_directory(
    TEST_DIR,
    image_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    label_mode='categorical'
)

class_names = train_ds.class_names
NUM_CLASES = len(class_names)
print(f"Clases detectadas: {class_names}")


train_ds = train_ds.map(lambda x, y: (preprocess_input(x), y))
val_ds = val_ds.map(lambda x, y: (preprocess_input(x), y))
test_ds = test_ds.map(lambda x, y: (preprocess_input(x), y))

AUTOTUNE = tf.data.AUTOTUNE
train_ds = train_ds.prefetch(buffer_size=AUTOTUNE)
val_ds = val_ds.prefetch(buffer_size=AUTOTUNE)


# Modelo
base_model = VGG16(
    weights='imagenet', 
    include_top=False, 
    input_shape=(224, 224, 3)
)

base_model.trainable = False

model = models.Sequential([
    base_model,
    layers.Flatten(),
    layers.Dense(256, activation='relu'),
    layers.Dropout(0.4), 
    layers.Dense(NUM_CLASES, activation='softmax') 
])

model.compile(
    optimizer=optimizers.Adam(learning_rate=1e-3),
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

print("Fase 1: Entrenando solo el clasificador superior")
history_fase1 = model.fit(train_ds, validation_data=val_ds, epochs=5)


# Fine Tuning
print("Fase 2: Configurando Fine-Tuning ---")

# Descongelamos el modelo base
base_model.trainable = True

# Volvemos a congelar todas las capas excepto las que tengan 'block5' en su nombre
for layer in base_model.layers:
    if 'block5' in layer.name:
        layer.trainable = True
        print(f"Capa {layer.name} lista para Fine-Tuning")
    else:
        layer.trainable = False

model.compile(
    optimizer=optimizers.Adam(learning_rate=1e-5),
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

# Entrenamos por más épocas para que los filtros convolucionales médicos se ajusten despacio
history_fase2 = model.fit(train_ds, validation_data=val_ds, epochs=12)


# Evaluacion
print("\n--- Evaluando modelo ajustado con el set de Prueba (Testing) ---")
loss, accuracy = model.evaluate(test_ds)
print(f"Precisión final en el set de prueba: {accuracy*100:.2f}%")

# Metricas 
x_test = []
y_true_encoded = []

for images, labels in test_ds:
    x_test.append(images.numpy())
    y_true_encoded.append(labels.numpy())

x_test = np.concatenate(x_test, axis=0)
y_true_encoded = np.concatenate(y_true_encoded, axis=0)

y_true = np.argmax(y_true_encoded, axis=1)

preds_probabilities = model.predict(x_test)
y_pred = np.argmax(preds_probabilities, axis=1)


print("\n" + "="*50)
print("         REPORTE DE CLASIFICACIÓN (VGG16)")
print("="*50)
print(classification_report(y_true, y_pred, target_names=class_names))
print("="*50)


# Matriz de confusion
cm = confusion_matrix(y_true, y_pred)

plt.figure(figsize=(8, 6))
sns.heatmap(
    cm, 
    annot=True, 
    fmt='d', 
    cmap='Blues', 
    xticklabels=class_names, 
    yticklabels=class_names
)
plt.title('Matriz de Confusión - Diagnóstico de Tumores Cerebrales')
plt.ylabel('Clase Real (Ground Truth)')
plt.xlabel('Clase Predicha por el Modelo')
plt.tight_layout()
plt.show()


acc_fase1 = history_fase1.history['accuracy']
val_acc_fase1 = history_fase1.history['val_accuracy']
loss_fase1 = history_fase1.history['loss']
val_loss_fase1 = history_fase1.history['val_loss']

# Extraemos las métricas de la Fase 2 (Fine-Tuning del Bloque 5)
acc_fase2 = history_fase2.history['accuracy']
val_acc_fase2 = history_fase2.history['val_accuracy']
loss_fase2 = history_fase2.history['loss']
val_loss_fase2 = history_fase2.history['val_loss']

# Concatenamos los arreglos para tener la secuencia completa de épocas
total_acc = acc_fase1 + acc_fase2
total_val_acc = val_acc_fase1 + val_acc_fase2
total_loss = loss_fase1 + loss_fase2
total_val_loss = val_loss_fase1 + val_loss_fase2

# Guardamos el número total de épocas y el punto exacto del cambio de fase
total_epochs = len(total_acc)
fase1_epochs = len(acc_fase1)


# Graficos de rendimiento
plt.figure(figsize=(14, 6))

# --- Gráfico 1: Precisión (Accuracy) ---
plt.subplot(1, 2, 1)
plt.plot(range(1, total_epochs + 1), total_acc, label='Entrenamiento (Train Acc)', color='royalblue', linewidth=2)
plt.plot(range(1, total_epochs + 1), total_val_acc, label='Validación (Val Acc)', color='orange', linewidth=2, linestyle='--')

# Línea vertical divisoria para marcar el inicio del Fine-Tuning
plt.axvline(x=fase1_epochs, color='red', linestyle=':', label='Inicio Fine-Tuning (Bloque 5)')

plt.title('Evolución de la Precisión (Accuracy) por Época', fontsize=12, fontweight='bold')
plt.xlabel('Épocas Totales', fontsize=10)
plt.ylabel('Exactitud (0.0 - 1.0)', fontsize=10)
plt.grid(True, linestyle='--', alpha=0.6)
plt.legend(loc='lower right')

# --- Gráfico 2: Pérdida (Loss) ---
plt.subplot(1, 2, 2)
plt.plot(range(1, total_epochs + 1), total_loss, label='Entrenamiento (Train Loss)', color='royalblue', linewidth=2)
plt.plot(range(1, total_epochs + 1), total_val_loss, label='Validación (Val Loss)', color='orange', linewidth=2, linestyle='--')

# Línea vertical divisoria para marcar el inicio del Fine-Tuning
plt.axvline(x=fase1_epochs, color='red', linestyle=':', label='Inicio Fine-Tuning (Bloque 5)')

plt.title('Evolución de la Función de Pérdida (Loss) por Época', fontsize=12, fontweight='bold')
plt.xlabel('Épocas Totales', fontsize=10)
plt.ylabel('Valor de Pérdida', fontsize=10)
plt.grid(True, linestyle='--', alpha=0.6)
plt.legend(loc='upper right')

plt.tight_layout()
# Opcional: Guarda la figura automáticamente para tu informe
plt.savefig('curvas_aprendizaje_vgg16.png', dpi=300)
plt.show()