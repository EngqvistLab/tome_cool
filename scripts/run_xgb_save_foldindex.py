#!/usr/bin/env python
# coding: utf-8

# #### 1. try different regression models
# ##### Gang Li, 2018-09-21

# In[2]:


import pandas as pd
import numpy as np
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import GridSearchCV
import sys
from sklearn.model_selection import KFold
from sklearn.metrics import r2_score as r2
import os
import xgboost as xgb

infile = sys.argv[1]
outdir = sys.argv[2]


def do_cross_validation(X,y,model,nfold=5):
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
        
        # train the model
        print(k,'Training')
        model.fit(X_train,y_train)
        
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
    
   

def test_model_performace(infile,outdir):
    df = pd.read_csv(infile,index_col=0)
    print(df.shape)
    X,y = df.values[:,:-1],df.values[:,-1]
    print(X.shape)
    
    model = xgb.XGBRegressor()
    name = 'XGBoost'
    print('Doing',name)
    results = do_cross_validation(X,y,model,nfold=5)
    outname = os.path.join(outdir,infile.split('/')[-1]+'_{0}.csv'.format(name.replace(' ','_')))
    save_results(df,results,outname)
    print('r2:', r2(results[:,0],results[:,1]))
    print(' ')
    
if __name__ == '__main__':
    #test_model_performace(infile,outfile)
    test_model_performace(infile,outdir)