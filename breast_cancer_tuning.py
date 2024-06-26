import pandas as pd
import keras
from keras.models  import Sequential
from keras.layers import Dense, Dropout
from keras.wrappers.scikit_learn import KerasClassifier
from sklearn.model_selection import GridSearchCV

previsores = pd.read_csv("entradas-breast.csv")
classe = pd.read_csv("saidas-breast.csv")

def criarRede(optimizer, loss, kernel_initializer, activation, neurons):
    #criando a rede neural
    classificador = Sequential()
    #adicionando a camada dense
    classificador.add(Dense(units = neurons,activation=activation,
                            kernel_initializer = kernel_initializer,input_dim = 30))
    #aplicando dropout na primeira layer
    classificador.add(Dropout(0.2))
    
    #add layer 2
    classificador.add(Dense(units = neurons,activation=activation,
                            kernel_initializer = kernel_initializer))
    #aplicando dropout na segunda layer
    classificador.add(Dropout(0.2))
    
    #camada de saida,
    classificador.add(Dense(units = 1, activation = 'sigmoid'))
    
    classificador.compile(optimizer = optimizer, loss =loss,
                           metrics = ['binary_accuracy'])
    return classificador

classificador = KerasClassifier(build_fn = criarRede)
parametros = {'batch_size': [10,30],
              'epochs':[50,100],
              'optimizer':['adam','sgd'],
              'loss': ['binary_crossentropy','hinge'],
              'kernel_initializer': ['random_uniform','normal'],
              'activation': ['relu','tanh'],
              'neurons': [16,8]}
grid_search = GridSearchCV(estimator=classificador,
                           param_grid=parametros,
                           scoring = 'accuracy',
                           cv = 5)

grid_search = grid_search.fit(previsores,classe)
melhores_parametros = grid_search.best_params_
melhor_precisao = grid_search.best_score_