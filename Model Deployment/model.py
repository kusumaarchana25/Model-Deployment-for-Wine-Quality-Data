#importing the libraries
import pickle
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression

#collecting the dataset
def load_data(file):
    df=pd.read_csv(file)

    #shape and size of the dataset
    print('shape of the dataframe: ',df.shape)
    print('size of the dataframe: ',df.size)

    #independent variables and target variable
    X=df.drop(columns=['quality'],axis=1)
    y=df['quality']
    #y = df['quality'].apply(lambda y_value: 1 if y_value >= 7 else 0)
    return X,y

#splitting the data into train and test
def split_data(X,y):
    X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.3,random_state=1)
    print('shape of X_train: ',X_train.shape)
    print('shape of X_test: ',X_test.shape)
    print('shape of y_train: ',y_train.shape)
    print('shape of y_test: ',y_test.shape)
    return X_train,X_test,y_train,y_test

#standardization scaler
def std_scaler(X_train,X_test):
    scaler=StandardScaler()
    X_train=scaler.fit_transform(X_train)
    X_train=pd.DataFrame(X_train)

    X_test=scaler.fit_transform(X_test)
    X_test=pd.DataFrame(X_test)
    return X_train,X_test

def main():

    print('starting of model file {}'.format(__name__))
    #path of the data file
    file=(r'data\winequality-red.csv')

    #loading the data
    X,y=load_data(file)

    #split the data into train and test
    X_train,X_test,y_train,y_test=split_data(X,y)

    #scaling the data
    X_train,X_test=std_scaler(X_train,X_test)

    #building the model
    linear_reg=LinearRegression()

    #fitting the model
    linear_reg.fit(X_train,y_train)

    #predicting the data
    y_pred=linear_reg.predict(X_test)

    #dumping the data to pickle file
    pickle.dump(linear_reg, open(r'models\model_dep.pkl','wb'))


    #for future purpose, we can load the pickle file
    LR_model=pickle.load(open(r'models\model_dep.pkl','rb'))

    #11 features
    print(LR_model.predict([[1,6,88,0.2,9,0.1,0.7,0.3,0,0.5,1.1]]))

    print('ending of model file {}'.format(__name__))

if __name__=="__main__":
    main()
