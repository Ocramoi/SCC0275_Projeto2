#!/usr/bin/env python3

## Imports

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import pandas as pd

## Setup

DATA_PATH = "./data/"

def getPath(_file: str) -> str:
    return DATA_PATH + _file

## Q1

_dfTrain = pd.read_csv(getPath("train.csv"))
_dfTrain

## Q1a

"""
A coluna que indica a classificação é a _Class_, indicando como booleano caso a entrada seja uma fraude.
"""

# Gera histpgrama da distribuição da coluna 'Class'
sns.histplot(_dfTrain['Class'])
# Utiliza escala logarítimica em y para melhor visualização
plt.yscale('log')
plt.show()

## Q1b

"""
A coluna de metadados é a coluna _Time_, que indica o timestamp em relação ao início da coleta apenas, não dados coletados.
"""

dfTrain = _dfTrain.drop(['Time'], axis=1)
dfTrain

## Q2

# Carrega as métricas desejadas
from sklearn.metrics import accuracy_score, roc_auc_score

# Data
X = dfTrain.drop(['Class'], axis=1)
# Targets
y = dfTrain['Class']

## Q2a

# Número de repetições para teste com o vetor randômico com metade de 1s e
# metade de 0s
N_RUNS = 30
CLASS_TYPES = [
    ['50/50', 0, 0],
    ['100%', 0, 0],
    ['0%', 0, 0],
]


artificial: np.ndarray = np.array([])

# Para estimador com 100% de fraudes
artificial = np.ones(y.shape)
CLASS_TYPES[1][1] = accuracy_score(y, artificial)
CLASS_TYPES[1][2] = roc_auc_score(y, artificial)

# Para estimador com 0% de fraudes
artificial = np.zeros(y.shape)
CLASS_TYPES[2][1] = accuracy_score(y, artificial)
CLASS_TYPES[2][2] = roc_auc_score(y, artificial)

# Para estimador com 50% de fraudes
for _ in range(N_RUNS):
    artificial = np.random.choice(a=[0,1], size=y.shape)
    CLASS_TYPES[0][1] += accuracy_score(y, artificial)/N_RUNS
    CLASS_TYPES[0][2] += roc_auc_score(y, artificial)/N_RUNS

for _TYPE in CLASS_TYPES:
    print(
        "Modelo {}:\n".format(_TYPE[0]),
        "\tAcurácia: {}\n".format(_TYPE[1]),
        "\tAUC: {}".format(_TYPE[2]),
        "\n"
    )

## Q2b

"""
Como podemos ver pelos dados obtidos, a métrica a ser adotada deve ser o AUC, já que o número de entradas da classe 1 é ordens de grandeza menor que as de class 0, assim um modelo que simplesmente classifica todas as entradas como 0 atinge uma acurácia de > 99%, enquanto o _score_ AUC se mantém estável para todos os conjunto artificiais gerados.
"""

## Q3

# Imports necessários
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from typing import Dict, List

## Q3a

# Conjunto de parâmetros para teste
paramGrid: Dict[str, List[int]] = {
    'n_estimators': [ 10, 50, 100, 200 ],
    'max_depth': [ 2, 3, 4, 5 ],
    'random_state': [ 42 ],
}

# Testa conjunto
rf = RandomForestClassifier()
clf = GridSearchCV(rf, paramGrid, cv=3, scoring='roc_auc')
clf.fit(X, y)
