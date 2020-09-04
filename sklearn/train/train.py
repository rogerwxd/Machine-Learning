import pandas as pd
from sklearn import tree
import os
from sklearn.ensemble import RandomForestClassifier, BaggingClassifier, AdaBoostClassifier
from sklearn.model_selection import train_test_split
import joblib
from sklearn.neural_network import MLPClassifier

saveModel = 'saveModel/' # PASTA PARA SALVAR O MODELO
caminho = 'datasetTrain/' # PASTA ONDE ESTÃO OS ARQUIVOS DE TRAINO
caminho_list = (os.listdir(caminho))
caminho_list.sort()


for data in caminho_list:

   dataset_nome = data.split('.')
   dataset_nome = dataset_nome[0]
   data_capture2 = caminho + data
   data_capture = pd.read_csv(data_capture2)
   x = data_capture.drop('class',axis=1)
   y = data_capture['class']

   # 70% dos dados para Train, 30% de dados para Test
   xTrain, xTest, yTrain, yTest = train_test_split(x, y, test_size = 0.30, random_state = 42)

   ############################################
   ##########      DECISION TREE     ##########
   ############################################
   # Making a decision tree with two levels.
   clfTre = tree.DecisionTreeClassifier(max_depth=None)
   clfTre.fit(xTrain, yTrain)
   score = clfTre.score(xTrain, yTrain)
   print("DECISION TREE: ", str(score))
   print('SAVE MODEL - DECISION TREE')
   scalerfile = saveModel + dataset_nome + '_decisionTree.pkl'
   joblib.dump(clfTre, scalerfile)
   print("DECISION TREE OK ")


   ############################################
   ##########      RANDOM FOREST      #########
   ############################################
   rf = RandomForestClassifier(n_estimators=100, max_depth=None, random_state=0, n_jobs=1)
   rf.fit(xTrain, yTrain)
   score = rf.score(xTrain, yTrain)
   print("RANDOM FLOREST: ", str(score))
   print('SAVE MODEL - RANDOM FOREST')
   scalerfile = saveModel + dataset_nome + '_randomForest.pkl'
   joblib.dump(rf, scalerfile)
   print("RANDOM FLOREST OK ")

   ############################################
   ###########      ADA BOOSTING ##############
   ############################################
   Adaclf = AdaBoostClassifier(n_estimators=100, random_state=0)
   Adaclf.fit(xTrain, yTrain)
   score = Adaclf.score(xTrain, yTrain)
   print("ADA BOOSTING ", str(score))
   print('SAVE MODEL - ADA BOOSTING')
   scalerfile = saveModel + dataset_nome + '_adaBoosting.pkl'
   joblib.dump(Adaclf, scalerfile)
   print("GRADIENT BOOSTING OK ")

   ############################################
   ##########      BAGGING       ##############
   ############################################
   bagClf = BaggingClassifier(n_estimators=100, random_state = 0, n_jobs=8)
   bagClf.fit(xTrain, yTrain)
   score = bagClf.score(xTrain, yTrain)
   print("BAGGING ", str(score))
   print('SAVE MODEL - BAGGING')
   scalerfile = saveModel + dataset_nome + '_bagging.pkl'
   joblib.dump(bagClf, scalerfile)
   print("BAGGING OK")

   ############################################
   ###############      MLP      $#############
   ############################################
   mlp = MLPClassifier(random_state=1, max_iter=300)
   mlp.fit(xTrain, yTrain)
   score = mlp.score(xTrain, yTrain)
   print("MLP ", str(score))
   print('SAVE MODEL - MLP')
   scalerfile = saveModel + dataset_nome + '_mlp.pkl'
   joblib.dump(mlp, scalerfile)
   print("MLP OK ")