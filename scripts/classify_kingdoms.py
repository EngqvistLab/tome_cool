#!/usr/bin/env python
# coding: utf-8

# In[13]:


import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_score
import os

# In[15]:


infile = '../data/dimer_updated_with_madin_bowman_kingdom.csv'
outdir = '../results/ml_classical_models/'

df = pd.read_csv(infile,index_col=0)

X = df.values[:,:-2]
y = df['domain'].values

print(X.shape,y.shape)

# In[18]:


def random_forest():
    parameters = {
                    'max_features':np.arange(0.1,1.1,0.1)
    }
    rf = RandomForestClassifier(n_estimators=1000)
    model=GridSearchCV(rf,parameters,n_jobs=-1,cv=3)
    return model


# In[19]:


def do_cross_validation(X,y,model):
    fea_impot = list()
    scores = list()
    results = np.zeros([len(y),3],dtype=str) # [true,pred,fold]
    
    kf = KFold(n_splits=5,shuffle=True, random_state=212)
    k = 0
    for train, test in kf.split(X):
        X_train, X_test, y_train, y_test = X[train], X[test], y[train], y[test]
        model.fit(X_train,y_train)
        scores.append(model.score(X_test,y_test))
        fea_impot.append(model.best_estimator_.feature_importances_)
        
        p = model.predict(X_test)
        
        results[test,0] = y_test
        results[test,1] = p
        results[test,2] = k
        
        k+=1
        
    return scores,fea_impot,results

def save_results(df,results,outname):
    dfres = pd.DataFrame(data=results,index=df.index,columns=['true','pred','kfold'])
    dfres.to_csv(outname)
# In[20]:


model = random_forest()
scores,fea_impot ,results = do_cross_validation(X,y,model)

# save cross validation results
save_results(df,results,os.path.join(outdir,infile.split('/')[-1]+'_kfold.csv'))

print('5-fold cross-validation accuracy:')
print(scores)
print(str(np.mean(scores))+','+str(np.std(scores))+'\n')

dfimp = pd.DataFrame(data=fea_impot,columns=df.columns[:-2])
dfimp.to_csv(os.path.join(outdir,infile.split('/')[-1]+'_feature_importances_cv5.csv'))


# In[ ]:




