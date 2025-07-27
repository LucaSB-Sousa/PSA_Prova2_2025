#!/usr/bin/env python3

import numpy as np
import pandas as pd
from sklearn import svm
from sklearn.model_selection import train_test_split
from avaliacao_desempenho_classificadores import metrics 

# -------------------- Função de Leitura --------------------
def leitura_radar(filename):
    print("Lendo arquivo CSV...")
    df = pd.read_csv(filename)
    print("Arquivo lido com sucesso.")
    
    # Escolha as características
    features = ["rcs", "range_sc", "vr", "azimuth_sc"]
    X = df[features].values
    y = df["label_id"].values


    print("Total de amostras carregadas:", len(y))
    return X, y

X, y = leitura_radar('dados_concatenados_short.csv')

# Divisão treino/teste
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# Classificador principal
print("Iniciando treinamento da SVM...")
s = svm.SVC() 

s.fit(X_train, y_train)
print("Treinamento concluído!")

y_pred = s.predict(X_test)

# -------------------- MÉTRICAS --------------------
M, accuracy, recall, specificity, precision, F1 = metrics(y_test, y_pred)


print('\n Matriz de Confusão:\n', M.astype(int))
print(f' Accuracy:     {accuracy:.4f}')
print(f' Recall:       {recall:.4f}')
print(f' Specificity:  {specificity:.4f}')
print(f' Precision:    {precision:.4f}')
print(f' F1 Score:     {F1:.4f}')
