import pandas as pd
import keras
from keras.models  import Sequential
from keras.layers import Dense, Dropout
from keras.wrappers.scikit_learn import KerasClassifier
from sklearn.model_selection import cross_val_score

previsores = pd.read_csv("entradas-breast.csv")
classe = pd.read_csv("saidas-breast.csv")

def criarRede():
    #criando a rede neural
    classificador = Sequential()
    #adicionando a camada dense, units = (numero de entradas + Num de saidas)/2
    classificador.add(Dense(units = 16, activation = 'relu', 
                        kernel_initializer = 'random_uniform', input_dim = 30))
    #aplicando dropout na primeira layer
    classificador.add(Dropout(0.2))
    
    #add layer 2
    classificador.add(Dense(units = 16, activation = 'relu', 
                        kernel_initializer = 'random_uniform'))
    #aplicando dropout na segunda layer
    classificador.add(Dropout(0.2))
    
    #camada de saida,
    classificador.add(Dense(units = 1, activation = 'sigmoid'))
    
    otimizador = keras.optimizers.Nadam(learning_rate=0.002, beta_1=0.9, beta_2=0.999)
    # otimizador adam(descida do gradiente),learning rate 0.001 com decaimento de 0.0001
   # otimizador = keras.optimizers.Adam(lr = 0.001, decay = 0.0001, clipvalue = 0.5)
    # loss (binary_crossentropy = calculo do erro para binarios)
    classificador.compile(optimizer = otimizador, loss = 'binary_crossentropy',
                      metrics = ['binary_accuracy'])
    return classificador


classificador = KerasClassifier(build_fn = criarRede,epochs =100, batch_size = 10)
classi = criarRede()
classificador_json = classi.to_json()
with open('classificador_breast.json','w') as json_file:
    json_file.write(classificador_json)


resultados = cross_val_score(estimator = classificador,
                             X = previsores, y = classe,
                             cv = 10, scoring = 'accuracy')
media = resultados.mean()
media #media das acc
desvio = resultados.std()
desvio #desvio, quanto maior o desvio, maior a prob de overfitting