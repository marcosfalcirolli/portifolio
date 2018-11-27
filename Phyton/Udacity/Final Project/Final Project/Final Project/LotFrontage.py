# -*- coding: utf-8 -*-
#Importando livrarias
import numpy as np
import pandas as pd

def importData():
    data_set = pd.read_csv('dataset/train.csv')
    submission_set = pd.read_csv('dataset/test.csv')
    return (data_set, submission_set)

def predictLotFrontage():
    data_set, submission_set = importData()
       
    #Adding an empty sales price column in the test data
    submission_set["SalePrice"] = ""
    submission_set["SalePrice"] = np.nan
    
    #Concating the train and test data
    df = pd.concat([data_set, submission_set])
    df = df.reset_index()
    df = df[['Id', 'LotArea', 'LotConfig', 'LotFrontage']]
    data_predict = df.loc[df['LotFrontage'].isnull()]
    df = df.loc[df['LotFrontage'].notnull()]
    df = df.drop(df[df.LotFrontage >200].index)
    df = df.drop(df[df.LotArea >30000].index)
    
    LotFrontage = df.LotFrontage
    LotConfig = pd.get_dummies(df['LotConfig'])
    df = df.drop('LotFrontage', axis = 1)
    df = pd.concat([df, LotConfig], axis=1)
    df = pd.concat([df, LotFrontage], axis=1)
    df = df.drop("Corner", axis = 1)
    df = df.drop("LotConfig", axis = 1)
    
    dataset = df
    X = dataset.iloc[:, 1:-1].values
    y = dataset.iloc[:, -1].values
    # Splitting the dataset into the Training set and Test set
    from sklearn.model_selection import train_test_split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.20, random_state = 0)
    # Fitting LinearRegression to the Training set
    from sklearn.linear_model import LinearRegression
    regressor = LinearRegression()
    regressor.fit(X_train, y_train)
    LotFrontage = data_predict.LotFrontage
    LotConfig = pd.get_dummies(data_predict['LotConfig'])
    data_predict = data_predict.drop("LotFrontage", axis = 1)
    data_predict = pd.concat([data_predict, LotConfig], axis=1)
    data_predict = pd.concat([data_predict, LotFrontage], axis=1)
    data_predict = data_predict.drop("Corner", axis = 1)
    data_predict = data_predict.drop("LotConfig", axis = 1)
    #data_predict = data_predict.drop(columns="LotFrontage", axis = 1)
    data_predict["LotFrontage"] = regressor.predict(data_predict.iloc[:, 1:-1].values)
    data_set, submission_set= importData()
    #Adding an empty sales price column in the test data
    submission_set["SalePrice"] = ""
    submission_set["SalePrice"] = np.nan
    #Concating the train and test data
    data = pd.concat([data_set, submission_set])
    data = data.reset_index()
    inde = data['LotFrontage'].loc[data['LotFrontage'].isnull()].index
    for i in inde:
        data.at[i,'LotFrontage']=data_predict.at[i,'LotFrontage']
    data[["Id", "LotFrontage"]].to_csv("LotFrontage.csv", sep=',')
    return data[["LotFrontage"]]
