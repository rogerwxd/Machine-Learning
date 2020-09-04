import pandas as pd
import os
from sklearn.metrics import confusion_matrix
import joblib

loadModel = 'loadModel/' # PASTA PARA SALVAR O MODELO
caminho = 'datasetTest/' # PASTA ONDE ESTÃO OS ARQUIVOS DE TRAINO
caminho_list = (os.listdir(caminho))
caminho_list.sort()

for data in caminho_list:

    dataset_nome = data.split('.')
    dataset_nome = dataset_nome[0]
    escrever = str(dataset_nome) + ';'
    escrever2 = str(dataset_nome) + ';'
    data_capture = pd.read_csv(caminho + dataset_nome + '.csv')
    xTest = data_capture.drop('class',axis=1)
    yTest = data_capture['class']

    ############################################
    ###### DECISION TREE #######
    ############################################
    print('DECISION TREE: START')
    dt = joblib.load(loadModel + str(dataset_nome) + '_decisionTree.pkl')
    predictions = dt.predict(xTest)
    cm_dt = confusion_matrix(yTest, predictions)
    TN = cm_dt[0][0]
    FP = cm_dt[0][1]
    FN = cm_dt[1][0]
    TP = cm_dt[1][1]
    AccN = TN/(TN+FP)
    AccA = TP/(TP+FN)
    print('TN: ' + str(TN)+' - FP: '+str(FP)+' - FN: '+str(FN)+' - TP: '+str(TP))
    print('AccTN: ' + str(AccN)+' - AccTP: '+str(AccA))
    print('DECISION TREE: END')
    print('')


    ############################################
    ###### RANDOM FOREST #######
    ############################################
    print('RANDOM FOREST: START')
    dt = joblib.load(loadModel + str(dataset_nome) + '_randomForest.pkl')
    predictions = dt.predict(xTest)
    cm_dt = confusion_matrix(yTest, predictions)
    TN = cm_dt[0][0]
    FP = cm_dt[0][1]
    FN = cm_dt[1][0]
    TP = cm_dt[1][1]
    AccN = TN/(TN+FP)
    AccA = TP/(TP+FN)
    print('TN: ' + str(TN)+' - FP: '+str(FP)+' - FN: '+str(FN)+' - TP: '+str(TP))
    print('AccTN: ' + str(AccN)+' - AccTP: '+str(AccA))
    print('RANDOM FOREST: END')
    print('')

    ############################################
    ###### ADA BOSTING #######
    ############################################
    print('ADA BOSTING: START')
    dt = joblib.load(loadModel + str(dataset_nome) + '_adaBoosting.pkl')
    predictions = dt.predict(xTest)
    cm_dt = confusion_matrix(yTest, predictions)
    TN = cm_dt[0][0]
    FP = cm_dt[0][1]
    FN = cm_dt[1][0]
    TP = cm_dt[1][1]
    AccN = TN/(TN+FP)
    AccA = TP/(TP+FN)
    print('TN: ' + str(TN)+' - FP: '+str(FP)+' - FN: '+str(FN)+' - TP: '+str(TP))
    print('AccTN: ' + str(AccN)+' - AccTP: '+str(AccA))
    print('ADA BOSTING: END')
    print('')

    ############################################
    ###### BAGGING #######
    ############################################
    print('BAGGING: START')
    dt = joblib.load(loadModel + str(dataset_nome) + '_bagging.pkl')
    predictions = dt.predict(xTest)
    cm_dt = confusion_matrix(yTest, predictions)
    TN = cm_dt[0][0]
    FP = cm_dt[0][1]
    FN = cm_dt[1][0]
    TP = cm_dt[1][1]
    AccN = TN/(TN+FP)
    AccA = TP/(TP+FN)
    print('TN: ' + str(TN)+' - FP: '+str(FP)+' - FN: '+str(FN)+' - TP: '+str(TP))
    print('AccTN: ' + str(AccN)+' - AccTP: '+str(AccA))
    print('BAGGING: END')
    print('')

    ############################################
    ###### MLP #######
    ############################################
    print('MLP: START')
    dt = joblib.load(loadModel + str(dataset_nome) + '_mlp.pkl')
    predictions = dt.predict(xTest)
    cm_dt = confusion_matrix(yTest, predictions)
    TN = cm_dt[0][0]
    FP = cm_dt[0][1]
    FN = cm_dt[1][0]
    TP = cm_dt[1][1]
    AccN = TN/(TN+FP)
    AccA = TP/(TP+FN)
    print('TN: ' + str(TN)+' - FP: '+str(FP)+' - FN: '+str(FN)+' - TP: '+str(TP))
    print('AccTN: ' + str(AccN)+' - AccTP: '+str(AccA))
    print('MLP: END')
    print('')
