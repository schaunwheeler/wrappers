# -*_ coding: utf-8 -*-
"""
Created on Fri Mar 28 16:58 2014

@authors:
    schaunwheeler
    paulmeinshausen
"""

import pandas as pd
import numpy as np
import sklearn.ensemble as sken
import sklearn.preprocessing as skpp
import multiprocessing
from sklearn.base import clone
try:
   import cPickle as pickle
except:
   import pickle

def pair_random(X, yname, suffix='____random'):

    rand_data = X.drop([yname], axis=1).copy()
    rand_data.columns = [x+suffix for x in rand_data.columns]
    rand_data = rand_data.apply(np.random.permutation, axis=0)
    newdata = pd.concat([X.copy(), rand_data], axis=1)

    return newdata

def model_fit(X, yname, func, transform=True, append_data=False,
              sort_importances=True):

    modeler = clone(func)
    if transform:
        scaler = skpp.StandardScaler().fit(X.copy())
        new_data = scaler.transform(X.copy())
        new_data = pd.DataFrame(new_data, columns=X.columns, index=X.index)
    else:
        new_data = X.copy()
        
    _ = modeler.fit(X=new_data.drop([yname], axis=1), y=new_data[yname])

    if append_data:
        modeler.original_data_ = X.copy()
    
    if transform:
        modeler.scaler = scaler

        if append_data:
            modeler.original_data_scaled_ = new_data
        
    if sort_importances:
        modeler.feature_importances_series_ = pd.Series(
        modeler.feature_importances_, index=[x for x in X.columns if x!=yname])
        modeler.feature_importances_series_.sort(ascending=False)

    return modeler

def model_predict(model, test_data, yname, append_data=True):

    if hasattr(model, 'scaler'):
        test_data_scaled_ = pd.DataFrame(
            model.scaler.transform(test_data.copy()), 
            columns=test_data.columns, index=test_data.index)
        new_data = test_data_scaled_
    else:
        new_data = test_data.copy()
       
    model_preds = new_data.loc[:,[yname]]
    model_preds['prediction'] = model.predict(new_data.drop([yname], axis=1))
    
    if hasattr(model, 'scaler'):    
        yname_loc = new_data.columns.tolist().index(yname)
        
        fake_data = pd.DataFrame(columns=new_data.columns, 
                                 index=model_preds.index)
        fake_data[yname] = model_preds[yname]
        model_preds[yname] = model.scaler.inverse_transform(
            fake_data)[:,yname_loc]
        
        fake_data[yname] = model_preds['prediction']
        model_preds['prediction'] = model.scaler.inverse_transform(
            fake_data)[:,yname_loc]
    
    if append_data:
        model.test_data_ = test_data.copy()
        model.predictions_ = model_preds.astype('float64')
        
        if hasattr(model, 'scaler'):
            model.test_data_scaled_ = test_data_scaled_

    return model

def model_simulate(model, yname, pname, n_inc=100, n_compare=100, data=None):
    
    if hasattr(model, 'original_data_'):
        sim_data = model.original_data_.copy()
    else:
        sim_data = data.copy()
            
    sim_vals = {}
    for p in pname:
        sims = [sim_data[pname].quantile(x) for x in np.linspace(0, 1, n_inc)]
        sims = np.random.choice(sims, n_inc*n_compare)
        sim_vals[pname] = sims
    
    inds = np.random.choice(sim_data.shape[0], n_inc*n_compare)
    
    new_data = sim_data.iloc[inds, :].copy()
    
    for p in pname:    
        new_data.loc[:,pname] = sim_vals[pname]
    
    preds = model_predict(model, new_data, yname)
    
    output = pd.DataFrame()
    output[yname] = preds.predictions_['prediction'].values

    for p in pname:
        output[pname] = sim_vals[pname]
    
    return output
    
def model_crossval(X, yname, func, fold_index, **kwargs):

    output = model_fit(X.loc[fold_index,:].copy(), yname=yname, func=func, 
        append_data=True, **kwargs)
    output = model_predict(output, X.loc[~fold_index,:].copy(), yname, **kwargs)

    return output

def model_kfold(X, yname, func, engine, nfolds=10, **kwargs):

    folds = np.random.choice(nfolds, X.shape[0])

    if engine=='multiprocessing':
        engine = multiprocessing.pool.Pool()
        fold_list = [engine.apply_async(model_crossval, 
            (X.copy(), yname, func, folds!=f), kwargs) for f in np.unique(folds)]
        engine.close()
        engine.join()
        fold_list = [output.get() for output in fold_list] 
    elif engine is not None:
        fold_list = engine.map(func=model_crossval, 
            input_list=[dict(X=X.copy(), yname=yname, func=func, fold_list=folds!=f,
                 **kwargs)]*nfolds, 
            verbose=True)
    else:
        fold_list = [model_crossval(X.copy(), yname, func, folds!=f, **kwargs) for 
            f in np.unique(folds)]

    return fold_list

def model_performance(kfold_output):
    
    preds = pd.concat([x.predictions_ for x in kfold_output])
    
    kfold_preds = pd.DataFrame(columns=['pearson', 'kendall', 'mean_abs_err',
                                        'median_abs_err'], index=[0])
    kfold_preds['pearson'] = preds.corr().iloc[0,1]
    kfold_preds['kendall'] = preds.corr(method='kendall').iloc[0,1]
    kfold_preds['mean_abs_err'] = preds.T.diff().abs().T.iloc[:,1].mean()
    kfold_preds['median_abs_err'] = preds.T.diff().abs().T.iloc[:,1].median()
   
    return kfold_preds

def get_tree_importances(data, yname, model, combine=False):

    if type(model) is not list:
        try:
            outputs = [est.feature_importances_ for hold in model.estimators_ 
                for est in hold]
        except:
            outputs = [est.feature_importances_ for est in model.estimators_]
    else:
        outputs = []
        
        for m in model:
            try:
                imps = [est.feature_importances_ for hold in m.estimators_ for est in hold]
            except:
                imps = [est.feature_importances_ for est in m.estimators_]
            
            imps = pd.DataFrame(np.vstack(imps), columns=[x for x in data.columns 
                if x!=yname])
            
            outputs.append(imps)
    
    out = pd.concat(outputs, ignore_index=True)
    
    return out

def importance_probabilities(imps, suffix='____random', nperms=500):
    
    features = [x for x in imps.columns if not x.endswith(suffix)]
    imps_f = imps.loc[:,features]
    output = pd.DataFrame(index=features)
    output['mean_importance'] = imps_f.mean()
    output['snr'] = imps_f.mean()/imps_f.std()
    output['type_s'] = (imps_f > 0).mean()
    
    random_probs = []
    for feat in features:
        reals = imps.loc[:,feat].values
        randoms = imps.loc[:,feat+suffix].values
        prob = 0
        for _ in range(nperms):
            rands = np.random.permutation(randoms)
            prob += (reals > rands).mean()
        random_probs.append(prob/nperms)
    
    output['prob_vs_random'] = pd.Series(random_probs, index=features)
    output = output.sort(columns='prob_vs_random', ascending=False)
    
    return output


def split_training_testing(data, split_col, label, drop_cols=None):

    if drop_cols is not None:
        output = data.drop(drop_cols, axis=1)
    
    training = output[split_col]==label    
    
    class split_data:
        training = output[training].drop([split_col], axis = 1)
        testing = output[~training].drop([split_col], axis = 1)

    return split_data

def quantile_lift(original, predicted, n=10, metric=sum, plot=False):
    q = 1.0/n
    qs = np.arange(0.0, 1.0+q, q)
    levels = [original.quantile(x) for x in qs[1:-1]]
    levels = [-np.inf] + levels + [np.inf]
    grouper = pd.cut(original, bins=levels, labels=qs[1:].astype(str))
    output= pd.DataFrame()
    output['original'] = original.groupby(grouper).apply(metric)
    output['predicted'] = predicted.groupby(grouper).apply(metric)
    if plot:
        return output.plot()
    else:
        return output
   