#!/usr/bin/env python
# coding: utf-8

# #### 1. try different regression models
# ##### Gang Li, 2018-09-21

# In[2]:


import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import GridSearchCV
from sklearn.linear_model import LinearRegression as LR
from sklearn import svm
from sklearn.linear_model import ElasticNetCV
from sklearn.tree import DecisionTreeRegressor
from sklearn.linear_model import BayesianRidge
import sys
from sklearn.model_selection import KFold
from sklearn.metrics import r2_score as r2
from scipy import stats
import os

import argparse


def normalize(X,args):
    X_n = np.zeros_like(X)
    k = 0
    for i in range(X.shape[1]):
        x = X[:,i]
        
        if i in args.ignored_norm_cols:
            X_n[:,k] = x
        elif np.var(x) == 0: None
        else: X_n[:,k] = (x-np.mean(x))/np.var(x)**0.5
        k+=1
    return X_n

def do_cross_validation(X,y,w,model,nfold=5):
    # X: input features, normalized
    # y: target
    #
    
    y = y.reshape([len(y),1])
    print('Initialize kFold')
    kf = KFold(n_splits=nfold,random_state=1,shuffle=True)
    
    results = np.zeros([len(y),3]) # [true,pred,fold]
    k = 0
    for train_ind, test_ind in kf.split(X):
        X_train, X_test = X[train_ind,:], X[test_ind,:]
        y_train, y_test = y[train_ind,:], y[test_ind,:]
        
        print(k,'Training')
        if w is not None: 
            w_train, w_test = w[train_ind], w[test_ind]
            model.fit(X_train,y_train,sample_weight=w_train.ravel())
        else: model.fit(X_train,y_train)
        
        print(k,'Test')
        p = model.predict(X_test)
        
        print(k,'Saving')
        results[test_ind,0] = y_test.ravel()
        results[test_ind,1] = p.ravel()
        results[test_ind,2] = k
        k+=1
    return results

def save_results(df,results,outname):
    dfres = pd.DataFrame(data=results,index=df.index,columns=['true','pred','kfold'])
    dfres.to_csv(outname)

def lr():
    return LR()




def elastic_net():
    return ElasticNetCV(n_jobs=20)



def bayesridge():
    model = BayesianRidge()
    return model




def svr():
    parameters={
                'C':np.logspace(-5,10,num=16,base=2.0),
                'epsilon':[0,0.01,0.1,0.5,1.0,2.0,4.0]
                }
    
    svr = svm.SVR(kernel='rbf')
    model = GridSearchCV(svr,parameters,n_jobs=-1,cv=3)
    return model



def tree():
    parameters={
                'min_samples_leaf':np.linspace(0.01,0.5,10)
                }
    dtr=DecisionTreeRegressor()
    model=GridSearchCV(dtr,parameters,n_jobs=-1,cv=3)
    return model



def random_forest():
    parameters = {
                    'max_features':np.arange(0.1,1.1,0.1)
    }
    rf = RandomForestRegressor(n_estimators=1000)
    model=GridSearchCV(rf,parameters,n_jobs=-1,cv=3)
    return model
   


def test_model_performace(args):
    df = pd.read_csv(args.infile,index_col=0)
    print(df.shape)
    
    if args.weighted:
        X,y,w = df.values[:,:-2],df.values[:,-2], df.values[:,-1]
    else:
        X,y = df.values[:,:-1],df.values[:,-1]
        w = None
    
    if args.x_boxcox: 
        for i in range(X.shape[1]): X[:,i] = stats.boxcox(X[:,i])[0]
    
    if args.y_boxcox: 
        y = stats.boxcox(y)[0]

    print(X.shape)
    X = normalize(X,args)
    print(X.shape)
    
    if len(args.models)==0: model_names = ['Linear', 'BayesRige', 'SVR', 'Tree', 'RandomForest']
    else: model_names = args.models
    models = {
        'Linear'       : lr(),
        'BayesRige'    : bayesridge(),
        'SVR'          : svr(),
        'Tree'         : tree(),
        'RandomForest' : random_forest()
    }
    
    for name in model_names:
        print('Doing',name)
        results = do_cross_validation(X,y,w,models[name],nfold=5)
        
        infile_name = args.infile.split('/')[-1]
        outname = os.path.join(args.outdir,infile_name+'_xbox{0}_ybox{1}_{2}.csv'.format(args.x_boxcox,args.y_boxcox,name))
        save_results(df,results,outname)
        print('r2:', r2(results[:,0],results[:,1]))
        print(' ')
        
        
if __name__ == '__main__':
    #test_model_performace(infile,outfile)
    parser = argparse.ArgumentParser(description='Corss-vadidation with classical models')
    parser.add_argument('--infile', 
                        help='Input csv file with features, and label in the last column',
                        metavar='')
    
    parser.add_argument('--models', 
                        nargs='+',
                        type=str,
                        help='Models to test',
                        default=[])
    
    parser.add_argument('--ignored_norm_cols', 
                        nargs='+',
                        type=int,
                        help='Column index that are not going to be normalized',
                        default=[])
    
    parser.add_argument('--x_boxcox', 
                        action='store_true',
                        help='If apply boxcox on features',
                        default=False)
    
    parser.add_argument('--y_boxcox', 
                        action='store_true',
                        help='If apply boxcox on response values',
                        default=False)
    
    parser.add_argument('--weighted', 
                        action='store_true',
                        help='If apply boxcox on response values',
                        default=False)
    
    parser.add_argument('--outdir', 
                        help='Output directory',
                        metavar='')
    
    args = parser.parse_args()
    print(args)
    
    test_model_performace(args)
    
    
    
    
    
    
    
    