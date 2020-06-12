# -*- coding: utf-8 -*-
"""
Created on Fri Jun 12 13:13:29 2020

@author: Alba Casillas Rodríguez y Francisco Javier Bolívar Expósito
"""


import numpy as np
import pandas as pd
import seaborn as sns
#import mlxtend
import matplotlib.pyplot as plt
import warnings 
warnings.filterwarnings("ignore")

from sklearn.neighbors import LocalOutlierFactor
from sklearn.decomposition import PCA
#from mlxtend.plotting import plot_learning_curves

from sklearn.linear_model import LogisticRegression
from sklearn.linear_model import Perceptron
from sklearn.linear_model import SGDClassifier
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
from sklearn.feature_selection import VarianceThreshold
from sklearn.svm import LinearSVC

from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import cross_val_score
from sklearn.metrics import classification_report






def leer_datos(archivo, separador=None):
    #Leemos los datos del archivo y lo guardamos en un Dataframe
    
    # NOTA:
    # Usando: pd.read_csv(archivo) , la primera fila al mostrar los datos se visualiza:
    # 0  1   6  15  12  1.1  0.1  ...  6.3  14.1  7.4  1.3  0.24  0.25  0.26
    # es decir, con números flotantes que no corresponden a los datos. Esto sucede porque con pandas
    # tiene que haber una primera fila que actúa como cabecera
    # He resuelto esta situacion consultando en: https://stackoverflow.com/questions/28382735/python-pandas-does-not-read-the-first-row-of-csv-file
    
    if separador == None:
        datos = pd.read_csv(archivo, header=None)
    else:
        datos = pd.read_csv(archivo, sep=separador , header=None)
    
    return datos


# Función que separa la muestra en sus características y etiquetas
    
def separar_datos(data):
    #Recoge los valores del dataframe
    valores = data.values
    
    #Todas las columnas menos la ultima
    X = valores[:, :-1]
    #La última columna
    Y = valores[:, -1]
    
    return X,Y




"""

    PROBLEMA DE CLASIFICACIÓN
            STATLOG
  
"""


###########################################################
#                                                         #
#                   MAIN DEL PROGRAMA                     #
#                                                         #
###########################################################

# Comenzamos leyendo los datos del problema

print("Leemos los datos: ")
data_train = leer_datos('datos/shuttle.trn', ' ')
data_test = leer_datos('datos/shuttle.tst', ' ')

print("Mostramos los datos de entrenamiento: ")
print(data_train)

"""
# rename columns
columns = {}
names = [(x, 'Var ' + str(x)) for x in data_train.columns]
for old, new in names:
    columns[old] = new
    
data_train = data_train.rename(columns=columns)
data_train = data_train.rename(columns={'Var 9': 'target'})

print(data_train)
"""

dist = data_train.iloc[:, -1].value_counts()
fontdict = {'fontsize': 18, 'weight' : 'bold'}
#plot
plt.bar(dist.index, dist)
# info
plt.title("Distribución de clases", fontdict=fontdict)
plt.ylabel("% de Ejemplos", fontdict=fontdict)
plt.xlabel("Clases", fontdict=fontdict);
# plot values
for i in dist.index:
    plt.text(i - 0.1, dist[i], dist[i], 
              fontsize=18)
    
#input("\n--- Pulsar tecla para continuar ---\n")