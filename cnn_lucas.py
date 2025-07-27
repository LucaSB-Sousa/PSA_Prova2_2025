#!/usr/bin/env python3

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
import tensorflow as tf
from tensorflow.keras import layers, models
import matplotlib.pyplot as plt
from avaliacao_desempenho_classificadores import metrics 


N_BINS_RANGE = 32
N_BINS_AZIMUTH = 32
RANGE_MIN, RANGE_MAX = 0, 100
AZIMUTH_MIN, AZIMUTH_MAX = -1, 1

# -------------------- CARREGAR OS DADOS ---------------------
df = pd.read_csv('dados_concatenados_short.csv')

frames = []
labels = []

for timestamp, group in df.groupby('timestamp'):
    grid = np.zeros((N_BINS_RANGE, N_BINS_AZIMUTH))
    for _, row in group.iterrows():
        i = int((row['range_sc'] - RANGE_MIN) / (RANGE_MAX - RANGE_MIN) * (N_BINS_RANGE - 1))
        j = int((row['azimuth_sc'] - AZIMUTH_MIN) / (AZIMUTH_MAX - AZIMUTH_MIN) * (N_BINS_AZIMUTH - 1))
        i = np.clip(i, 0, N_BINS_RANGE - 1)
        j = np.clip(j, 0, N_BINS_AZIMUTH - 1)
        grid[i, j] = row['rcs']
    frames.append(grid)
    labels.append(group['label_id'].mode()[0])

X = np.array(frames)[..., np.newaxis]
y = np.array(labels)

print("Shape X:", X.shape, "Shape y:", y.shape)
print("Classes únicas:", np.unique(y))

unique_classes = np.unique(y)
if len(unique_classes) < 2:
    raise ValueError(f"O conjunto de dados tem apenas uma classe ({unique_classes[0]}).")

# -------------------- SPLIT TREINO/TESTE ---------------------
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)


num_classes = int(np.max(y) + 1)

# -------------------- DEFINIÇÃO E TREINO DA CNN ---------------------
model = models.Sequential([
    layers.Input(shape=(N_BINS_RANGE, N_BINS_AZIMUTH, 1)),
    layers.Conv2D(16, (3, 3), activation='relu'),
    layers.MaxPooling2D((2, 2)),
    layers.Conv2D(32, (3, 3), activation='relu'),
    layers.MaxPooling2D((2, 2)),
    layers.Flatten(),
    layers.Dense(64, activation='relu'),
    layers.Dense(num_classes, activation='softmax')
])

model.compile(
    optimizer='adam',
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy']
)

model.summary()

history = model.fit(
    X_train, y_train,
    epochs=15,
    batch_size=32,
    validation_data=(X_test, y_test)
)

# -------------------- AVALIAÇÃO E VISUALIZAÇÃO ---------------------
plt.plot(history.history['accuracy'], label='Treino')
plt.plot(history.history['val_accuracy'], label='Validação')
plt.xlabel('Epoch')
plt.ylabel('Acurácia')
plt.legend()
plt.title('Histórico de Acurácia')
plt.grid(True)
plt.show()

test_loss, test_acc = model.evaluate(X_test, y_test, verbose=2)
print('\n✅ Acurácia no conjunto de teste:', test_acc)

# -------------------- MÉTRICAS ---------------------
y_pred_probs = model.predict(X_test)
y_pred_classes = np.argmax(y_pred_probs, axis=1)


M, accuracy, recall, specificity, precision, F1 = metrics(y_test, y_pred_classes)

# Exibe os resultados
print('\n Matriz de Confusão:\n', M.astype(int))
print(f' Accuracy:     {accuracy:.4f}')
print(f' Recall:       {recall:.4f}')
print(f' Specificity:  {specificity:.4f}')
print(f' Precision:    {precision:.4f}')
print(f' F1 Score:     {F1:.4f}')
