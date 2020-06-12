# -*- coding: utf-8 -*-
"""
PROBLEMA DE CLASIFICACIÓN - STATLOG

Autores: Alba Casillas Rodríguez y Francisco Javier Bolívar Expósito
"""

import warnings
import numpy as np
import pandas as pd

import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import PolynomialFeatures
from sklearn.model_selection import GridSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier

# Establecemos semilla para obtener resultados reproducibles
np.random.seed(500)


# Lectura de un conjunto de muestraas
def read_data(path, delim=' ', dtype=np.int32):
    data_set = np.loadtxt(path, dtype, None, delim)

    x = data_set[:, :-1]
    y = np.ravel(data_set[:, -1:])

    return x, y

def plot_class_distribution(labels, set_title):
    dist = np.array(np.unique(labels, return_counts=True)).T
    # Barplot
    sns.countplot(labels)
    # Legends
    plt.title(set_title + ' Class Distribution')
    plt.ylabel("Number of Samples")
    plt.xlabel("Class");
    # Values
    for i in dist[:, 0]:
        plt.text(i - 1, dist[i - 1, 1], dist[i - 1, 1])

    plt.show()

###########################################################
#                                                         #
#                   MAIN DEL PROGRAMA                     #
#                                                         #
###########################################################

# Comenzamos leyendo los datos del problema

print("Leemos los datos")
x_train, y_train = read_data('./datos/shuttle.trn')
x_test, y_test = read_data('./datos/shuttle.tst')

# Análisis del problema
# Estadísticas sobre las características
print(pd.DataFrame(x_train).describe().to_string())

#input("\n--- Pulsar tecla para continuar ---\n")

# Comprobamos si existen datos perdidos en el dataset
print("¿Existen valores perdidos?: ", end='')
print(pd.DataFrame(np.vstack([x_train, x_test])).isnull().values.any())


#input("\n--- Pulsar tecla para continuar ---\n")

# Vemos la distribución de clases tanto en el train set como en el test set
print("Distribución de clases para cada conjunto:")
plot_class_distribution(y_train, 'Training Set')
plot_class_distribution(y_test, 'Test Set')

#input("\n--- Pulsar tecla para continuar ---\n")

print("Matriz de correlación entre los atributos:")
matrix_corr = pd.DataFrame(x_train).corr('pearson').round(3)
sns.heatmap(matrix_corr, annot = True)
plt.show()

#input("\n--- Pulsar tecla para continuar ---\n")

# Preprocesamiento de datos
x_train_pol = PolynomialFeatures().fit_transform(x_train)
x_test_pol = PolynomialFeatures().fit_transform(x_test)

# Normalización
x_train = StandardScaler(copy=False).fit_transform(x_train)
x_train_pol = StandardScaler(copy=False).fit_transform(x_train_pol)
x_test = StandardScaler(copy=False).fit_transform(x_test)
x_test_pol = StandardScaler(copy=False).fit_transform(x_test_pol)

# Selección de modelo y entrenamiento
# Se eligen los mejores hiperparámetros para los modelos 'LogisticRegression' y
# 'logRegPol' usando validación cruzada 5-fold partiendo el train set,print('LR Train-Accuracy: ' + str(ein_reg))

# tras esto se entrena cada modelo usando todo el train set.
parameters_log = [{'penalty': ['l1', 'l2'], 'C': np.logspace(-3, 3, 7)}]
columns_log = ['mean_fit_time', 'param_C', 'param_penalty', 'mean_test_score',
               'std_test_score', 'rank_test_score']

parameters_rf = [{'n_estimators' : [10,100,250,500] , 'max_features': ['auto', 'sqrt', 'log2']}]
columns_rf = ['mean_fit_time', 'mean_test_score', 'mean_score_time', 'std_score_time', 'param_max_features', 'param_n_estimators']                                        


#logReg = GridSearchCV(LogisticRegression(solver='saga'), parameters_log)
#logReg.fit(x_train, y_train)
#logRegPol = GridSearchCV(LogisticRegression(solver='saga'), parameters_log)
#logRegPol.fit(x_train_pol, y_train)
randomForest  = GridSearchCV(RandomForestClassifier(), parameters_rf, n_jobs = -1)
randomForest.fit(x_train, y_train)
#print('CV para RL\n', pd.DataFrame(logReg.cv_results_, columns=columns_log).to_string())
#print('CV para RL con combinación no lineal\n',
        # pd.DataFrame(logRegPol.cv_results_, columns=columns_log).to_string())
print('CV para RF\n', 
      pd.DataFrame(randomForest.cv_results_, columns=columns_rf).to_string())

# Se muestran los hiperparámetros escogidos y Eval para ambos modelos
# Observamos que la Regresión Logística proporciona mejores resultados
# print('\nResultados de selección de hiperparámetros por validación cruzada')
# print("LR Best hyperparameters: ", logReg.best_params_)
# print("LR CV-Accuracy :", logReg.best_score_)

# print("LRP Best hyperparameters: ", logRegPol.best_params_)
# print("LRP CV-Accuracy :", logRegPol.best_score_)

print("RF Best hyperparameters : ", randomForest.best_params_)
print("RF CV-Accuracy :", randomForest.best_score_)

input("\n--- Pulsar tecla para continuar ---\n")

# # Predicción con los modelos entrenados del train y test set
# print('Métricas de evaluación para los modelos entrenados para train y test')
# ein_reg = logReg.score(x_train, y_train)
# ein_lrp = logRegPol.score(x_train_pol, y_train)
ein_lrp = randomForest.score(x_train, y_train)
# print('LR Train-Accuracy: ' + str(ein_reg))
# print('LRP Train-Accuracy: ' + str(ein_lrp))
print('RF Train-Accuracy: ' + str(ein_lrp))

# etest_reg = logReg.score(x_test, y_test)
# etest_per = logRegPol.score(x_test_pol, y_test)
etest_rf = randomForest.score(x_test, y_test)
# print('\nLR Test-Accuracy: ' + str(etest_reg))
# print('LRP Test-Accuracy: ' + str(etest_per))
print('RF Test-Accuracy: ' + str(etest_rf))
