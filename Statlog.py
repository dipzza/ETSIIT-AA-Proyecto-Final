# -*- coding: utf-8 -*-
"""
PROBLEMA DE CLASIFICACIÓN - STATLOG

Autores: Alba Casillas Rodríguez y Francisco Javier Bolívar Expósito
"""


import numpy as np
import pandas as pd

import matplotlib.pyplot as plt
import seaborn as sns

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

print("Leemos los datos: ")
x_train, y_train = read_data('./datos/shuttle.trn')
x_test, y_test = read_data('./datos/shuttle.tst')

# Análisis del problema

# Vemos la distribución de clases tanto en el train set como en el test set
plot_class_distribution(y_train, 'Training Set')
plot_class_distribution(y_test, 'Test Set')

    
#input("\n--- Pulsar tecla para continuar ---\n")
    


#input("\n--- Pulsar tecla para continuar ---\n")
    
# Preprocesamiento de datos

# matrix_corr = pd.DataFrame(X_train).corr('pearson')
# sns.heatmap(matrix_corr, annot = True)
# plt.show()