#Kernel base: https://www.kaggle.com/vforvince1/visualizing-data-and-predicting-pokemon-fights

from itertools import combinations

import numpy as np

from subprocess import check_output
#print(check_output(["ls", "./dataset"]).decode("utf8"))

import pandas as pd         #http://pandas.pydata.org/pandas-docs/stable/io.html

pokemons = pd.read_csv('./dataset/pokemon.csv')
combates = pd.read_csv('./dataset/combats.csv')
combates_teste = pd.read_csv('./dataset/tests.csv')

#print(pokemons.size, pokemons.shape)
#print(combates.size, combates.shape)
#print(combates_teste.size, combates_teste.shape)

#print(pokemons.Name)
#print(pokemons.Name[0:5])
#print(pokemons.head())
#print(combates.head())
#print(combates.head(5))

combates.Winner[combates.Winner == combates.First_pokemon] = 0
combates.Winner[combates.Winner == combates.Second_pokemon] = 1

x_treino_full = combates.drop('Winner', axis=1)
y_treino_full = combates.Winner


#http://scikit-learn.org/stable/modules/generated/sklearn.model_selection.train_test_split.html
from sklearn.model_selection import train_test_split
#test_size - Valor entre 0 e 1 que será a proporção da lista de Teste
x_treino, x_teste, y_treino, y_teste = train_test_split(x_treino_full, y_treino_full, test_size=0.25)

#print(x_treino.head()  )
#print(y_treino.head() )


from sklearn.tree import DecisionTreeClassifier
#min_samples_leaf - Quantidade mínima de amostras necessárias para ser uma folha
#min_samples_split - Quantidade mínima de amostras necessárias para dividir um nó
clf = DecisionTreeClassifier(random_state=0, min_samples_leaf= 20, min_samples_split=50)
model = clf.fit(x_treino, y_treino)
valor = model.score(x_teste, y_teste)
print( "Precisão por Árvore de Decisão: ", valor )