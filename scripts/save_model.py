'''
Gang Li
2019-01-23
'''

import os
import numpy as np
from sklearn.metrics import mean_squared_error as MSE
from sklearn.metrics import r2_score
from scipy.stats import spearmanr,pearsonr
from sklearn.model_selection import GridSearchCV
from sklearn.externals import joblib
import pandas as pd
from sklearn.linear_model import BayesianRidge
from sklearn.model_selection import KFold

def Normalize(X):
    mean,var=list(),list()
    for i in range(X.shape[1]):
        mean.append(np.mean(X[:,i]))
        var.append(float(np.var(X[:,i]))**0.5)
    return mean,var

def standardize(X):
    Xs=np.zeros_like(X)
    n_sample,n_features=X.shape[0],X.shape[1]
    for i in range(n_features):
        Xs[:,i]=(X[:,i]-np.mean(X[:,i]))/float(np.var(X[:,i]))**0.5
    return Xs


def bayesridge():
    model = BayesianRidge()
    return model

def CrossValidation(X,Y,w,f,n_folds):
    kf = KFold(n_splits=n_folds)
    
    p=np.zeros_like(Y)

    for train,test in kf.split(X):
        f.fit(X[train],Y[train],sample_weight=w[train])
        try:print(f.best_params_)
        except: None
        p[test]=f.predict(X[test])
    rmse_cv=np.sqrt(MSE(Y,p))
    r2_cv=r2_score(Y,p)
    r_spearman=spearmanr(Y,p)
    r_pearson=pearsonr(Y,p)
    return p,rmse_cv,r2_cv,r_spearman[0],r_pearson[0]

infile ='../data/dimer_updated_with_madin_bowman_uniform_weights.csv'
outdir = '../tome_cool/'

if not os.path.exists(outdir): os.mkdir(outdir)

report=open(os.path.join(outdir,'report.txt'),'w')

# data
df = pd.read_csv(infile,index_col=0)
X = df.values[:,:-2]
Y = df.values[:,-2].ravel() 
w = df.values[:,-1].ravel() 

Xs = standardize(X)
features = df.columns[:-2]

# train model
model = bayesridge()
model.fit(Xs,Y,sample_weight=w)

# Model stats
p = model.predict(Xs)
rmse_cv = np.sqrt(MSE(p,Y))
r2_cv = r2_score(Y,p)
r_spearman = spearmanr(p,Y)
r_pearson = pearsonr(p,Y)
res = 'rmse:{:.4}\nr2:{:.4}\nspearmanr:{:.4}\np_value:{:.4}\npearonr:{:.4}\np_pearsonr:{:.4}'.format(rmse_cv,r2_cv,r_spearman[0],r_spearman[1],r_pearson[0],r_pearson[1])
print(res)
report.write(res+'\n')


# Save the predicted results in a flat file
outf=open(os.path.join(outdir,'training_results.csv'),'w')
outf.write('index,exprimental,predicted\n')
for i in range(Xs.shape[0]):
    outf.write('{},{},{}\n'.format(df.index[i],Y[i],p[i]))
outf.close()
# Save model with joblib

predictor_dir = os.path.join(outdir,'predictor')
if not os.path.exists(predictor_dir): os.mkdir(predictor_dir)

model_name = os.path.join(predictor_dir,'model')
joblib.dump(model,model_name+'.pkl')
fea = open(model_name+'.f','w')
mean,var = Normalize(X)

print('length of means:',len(mean))
print('length of vars:',len(var))
fea.write('#Feature_name\tmean\tsigma\n')
for i in range(len(mean)):fea.write('{}\t{}\t{}\n'.format(features[i],mean[i],var[i]))
fea.close()
