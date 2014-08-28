# -*_ coding: utf-8 -*-
"""
Created on Fri Mar 28 16:58 2014

@authors:
    schaunwheeler
"""

import pandas as pd
import numpy as np
import copy
from sklearn.base import clone
from sklearn.linear_model import LinearRegression
import fastcluster
from scipy.spatial.distance import squareform
from scipy.cluster.hierarchy import dendrogram, fcluster
from sklearn.metrics.pairwise import pairwise_distances
import statsmodels.nonparametric.smoothers_lowess as smoother
import time

def geometric_mean(x):
    return np.exp(np.log(1+np.abs(x)).mean())-1

def progress_bar(item, l, interval=0.05):
    rounder = len(str(interval).split('.')[1])
    prange = np.arange(0, 1+interval, interval)
    prange = [round(x, rounder) for x in prange]
    n = len(l)
    i = l.index(item)
    if (i>0.0) & (i<(n-1)):
        old_percent = (i-1)/n
        percent = i/n
        prange = [x for x in prange if x>old_percent]
        if percent>=min(prange):
            print percent,
    else:
        print i/(n-1),

def linear_impute(x, poly=None, log=False):
    y = x.copy()        
    if log:
        y = np.log(y+1)
    to_test = y.isnull()
    train = pd.DataFrame({'x':y[~to_test].index.values})
    test = pd.DataFrame({'x':[y[to_test].index.values]})
    if poly is not None:
        for p in poly:
            train['x'+str(p)] = train['x']**p
            test['x'+str(p)] = test['x']**p
    f = LinearRegression().fit(X=train, y=y.dropna())
    out = f.predict(test)
    y[y.isnull()] = out[0]
    if log:
        y = np.exp(y)-1
    return y


def simple_lag_estimate(x, shift, poly=None, log=False, index=None):
    if type(index)==str:
        index = x.index.get_level_values(index).values
    elif index is not None:
        y = pd.Series(x, index=index[:len(x)])
    else:
        y = x.copy()        
    if log:
        y = np.log(y+1)
    train = pd.DataFrame({'x':y.index.values})
    target = y.index.values.max()+shift
    test = pd.DataFrame({'x':[target]})
    if poly is not None:
        for p in poly:
            train['x'+str(p)] = train['x']**p
            test['x'+str(p)] = test['x']**p
    f = LinearRegression().fit(X=train, y=y.dropna())
    out = f.predict(test)[0]
    if log:
        out = np.exp(out)-1
    out = pd.DataFrame(out, columns=[target], index=x.columns).T

    return out 

def lag_data(data, yname, lags, time_var, append=None, group=None, extra=None, 
             linear_guess=None):
    
    index_vars = []
    if append is not None:
        index_vars = index_vars + append.keys()
        grouper = append.keys()
    if group is not None:
        index_vars = index_vars + group
        grouper = grouper + group
    if extra is not None:
        index_vars = index_vars + extra
    
    data = data.sort(time_var)
    data = data.set_index(index_vars+[time_var])
    
    data_list = []    
    for l in lags:
        print l, 
        
        main = data.groupby(level=index_vars).shift(l).dropna()
        if append is not None:
            y_list = {k:main[v].copy() for k,v in append.items()}
            drop_cols = list(set([x for y in append.values() for x in y]))
            main = main.drop(drop_cols, axis=1)
        
        if linear_guess is not None:
            y_lin = data[yname].copy()
            y_lin.index = y_lin.index.droplevel(extra)
            y_lin = y_lin.unstack(grouper)
            impute = y_lin.isnull().sum()>0
            y_lin.loc[:,impute] = y_lin.loc[:,impute].apply(
                lambda x: linear_impute(x, log=True))
            last_period = y_lin.index.values.max() - l
            periods = y_lin.loc[:last_period].index.values
            guess_df = pd.DataFrame(columns=periods[1:]+l, index=y_lin.columns)
            guess = pd.Series(np.zeros(y_lin.shape[1]), index=y_lin.columns)
            for p in periods[1:]:
                name = p+l
                for g in linear_guess:
                    keep = y_lin.loc[:p,:]
                    est = simple_lag_estimate(keep, shift=l, **g)
                    guess = guess + est.T.squeeze()
                guess = guess / len(linear_guess)
                guess_df[name] = guess
            guess_df.columns.names = [time_var]
            guess_df = guess_df.stack()
            matcher_y = pd.match(main.index.droplevel(extra).values,
                   guess_df.index.values)
            main_idx = matcher_y>=0
            matcher_y = matcher_y[main_idx]
            guess_df = guess_df.iloc[matcher_y]
            guess_df.index = main.index[main_idx]
            guess_df = guess_df.to_frame('guess')
            main = pd.concat([main, guess_df], axis=1)

        if append is not None:
            for k in append.keys():
                for c in y_list[k].columns:
                    y = y_list[k][c].copy()
                    y.index = y.index.droplevel(extra)
                    y = y.groupby(level=[time_var]+group, group_keys=False, 
                        sort=False).apply(lambda x: x.unstack(k))
                    y.columns = [c+'_'+str(x) for x in y.columns]
                    y = y.fillna(0)
                    matcher_y = pd.match(main.index.droplevel(extra+[k]).values,
                                       y.index.values)
                    main_idx = matcher_y>=0
                    matcher_y = matcher_y[main_idx]
                    y = y.iloc[matcher_y,:]
                    y.index = main.index[main_idx]
                    main = pd.concat([main, y], axis=1)

        main = main.dropna()
        data_list.append(main)

    data = data[[yname]]

    if lags != [0]:
        data = pd.concat([data]+data_list, axis=1, keys=[0]+lags,
                 names=['lag', 'variable'])
    else:
        data = pd.concat([data]+data_list, axis=1)
        if type(data.columns) == pd.MultiIndex:
            idx = pd.MultiIndex.from_tuples([(0,)+x for x in data.columns],
                names=['lag', 'variable'])
        else:
            idx = pd.MultiIndex.from_tuples([(0,x) for x in data.columns],
                 names=['lag', 'variable'])
        data.columns = idx

    keep_years = np.unique(data.dropna().index.get_level_values(time_var).values)

    if lags == [0]:
        keep_years = [np.max(keep_years)]
    keep_years = data.index.get_level_values(time_var).isin(keep_years)
    data = data[keep_years]
    
    data = data.fillna(0)
    
    return data
        
class Cluster(object):
    
    def __init__(self, data, value_var, cluster_vars, index_vars=None, 
                 scale=False):
        self.index_cols = cluster_vars[:]
        self.cluster_cols = cluster_vars[:]
        self.value_var = value_var
        self.threshold = None
        if index_vars is not None:
            self.index_cols += index_vars[:]
        dat = data.copy()
        dat = dat.set_index(self.index_cols)
        dat = dat[self.value_var].unstack(self.cluster_cols)
        self.original_data = data.copy()
        if scale:
            dat = (dat - dat.mean())/dat.std()
            self.original_data_scaled = dat.copy()
        self.data = dat
        
    def corr_dist(self, method='pearson', weight=True, auto_exclude=None,
                  absolute=False, scale=False, transform=None):

        data = self.data.copy()
        if transform is not None:
            data = data.apply(transform)
        
        if scale:
            data = (data - data.mean())/data.std()
        corrs = data.corr(method)
        if weight:
            jaccard = pairwise_distances(self.data.T.notnull().values, 
                                         metric='jaccard')
            jaccard = 1 - jaccard
            corrs = corrs * jaccard
            filler = 1.0
        else:
            filler = 0.5
                
        if type(corrs.index) == pd.core.index.MultiIndex:
            corrs.sortlevel(axis=0, inplace=True)
            corrs.sortlevel(axis=1, inplace=True)
        
        if absolute:
            diffs = 1.0-corrs.abs()
        else:
            diffs = (1.0-corrs)/2.0

        diffs = diffs.fillna(filler)
        diffs[diffs>1.0] = 1.0
        diffs[diffs<0.0] = 0.0
        
        if auto_exclude is not None:            
            for excl in auto_exclude: 
                groupings = diffs.columns.get_level_values(excl)
                group_mat = pd.get_dummies(groupings)
                group_mat = group_mat.dot(group_mat.T)
                diffs.values[~group_mat.astype('bool').values] = 1.0
            
        diffs = squareform(diffs.values, checks=False)
        
        self.correlation_matrix = corrs
        self.distance_array = diffs

    def linkage(self, method='complete', **kwargs):
        clust = fastcluster.linkage(self.distance_array, method=method, 
                                    **kwargs)
        self.linkage_output = clust
    
    def assign_clusters(self, t=0.0525, criterion='distance', **kwargs):
        fclust = fcluster(self.linkage_output, t=t, criterion=criterion, 
                          **kwargs)
        self.threshold = t
        self.cluster_assignments = fclust
        self.assignment_dict_keylevels = self.correlation_matrix.index.names
        self.assignment_dict = pd.Series(self.cluster_assignments, 
                index=self.correlation_matrix.index).to_dict()

    def dendrogram(self, plot=True, **kwargs):
        no_plot = not plot
        dend = dendrogram(self.linkage_output, 
            labels=self.correlation_matrix.columns, 
            color_threshold=self.threshold, no_plot=no_plot, **kwargs)
        self.dendrogram_dict = dend
    
    def collect_groups(self):
        dfs = []
        for var in self.cluster_cols:
            var_values = self.correlation_matrix.columns.get_level_values(var)
            groups = pd.Series(var_values).groupby(self.cluster_assignments)
            df = groups.apply(lambda x: x.tolist()).to_frame(var)    
            dfs.append(df)    
        output = pd.concat(dfs, axis=1)       
        output.index.name = 'cluster'
        output = output.reset_index().set_index('cluster', drop=False)
        self.groups = output
        
    def summarize_singleindex(self):
        output = self.groups.copy()
        var = self.cluster_cols[0]
        output['counts'] = output[var].apply(len)
        output['min_corr'] = np.nan
        output['mean_corr'] = np.nan
        output['max_corr'] = np.nan
        for c in np.unique(self.cluster_assignments):
            keep = self.correlation_matrix.columns.get_level_values(var)
            keep = keep.isin(output[var][c])
            c_corr = self.correlation_matrix.loc[keep, keep]
            c_corr_inds = np.triu_indices_from(c_corr, k=1)
            c_corr = c_corr.values[c_corr_inds]
            if len(c_corr)==0:
                c_corr = np.array([1])
            output['mean_corr'][c] = c_corr.mean()
            output['min_corr'][c] = c_corr.min()
            output['max_corr'][c] = c_corr.max()
        self.summary = output

    def summarize_multiindex(self):
        output = self.groups.copy()
        for var in self.cluster_cols:
            output[var+'_counts'] = output[var].apply(len)
            output[var+'_min_corr'] = np.nan            
            output[var+'_mean_corr'] = np.nan            
            output[var+'_max_corr'] = np.nan
            for c in np.unique(self.cluster_assignments):
                keep = self.correlation_matrix.columns.get_level_values(var)
                keep = keep.isin(output[var][c])
                c_corr = self.correlation_matrix.loc[keep, keep]
                c_corr_inds = np.triu_indices_from(c_corr, k=1)
                c_corr = c_corr.values[c_corr_inds]
                if len(c_corr)==0:
                    c_corr = np.array([1])
                output[var+'_mean_corr'][c] = c_corr.mean()
                output[var+'_min_corr'][c] = c_corr.min()
                output[var+'_max_corr'][c] = c_corr.max()
        self.summary = output

    def append_cluster_col(self, tag=None):
        if tag is None:
            tag = 'cluster_id'
            
        clust_data = pd.Series(self.assignment_dict)
        clust_data.index.names = self.assignment_dict_keylevels
        clust_data = clust_data.to_frame(tag).reset_index()
        new_data = pd.merge(self.original_data, clust_data, how='left', 
                on=self.cluster_cols)
        return new_data

    def group_recursive(self, corr_method='pearson', weight=True, absolute=False,
        auto_exclude=None, linkage_method='complete', linkage_kwargs=None,
        t=0.0525, criterion='distance', scale=False, transform=None, 
        **fclust_kwargs):
        if linkage_kwargs is None:
            linkage_kwargs = {}
        self.corr_dist(method=corr_method, weight=weight, absolute=absolute,
                       auto_exclude=auto_exclude, scale=scale, 
                       transform=transform)
        self.linkage(method=linkage_method, **linkage_kwargs)
        self.assign_clusters(t=t, criterion=criterion, **fclust_kwargs)
        if auto_exclude is None:
            keep = []
        else:
            keep = auto_exclude
        
        i=0
        old_tag = 'cluster_id'
        tag = 'cluster_id_'+str(i)
        n_clusts = np.inf
        new_data = self.append_cluster_col(tag=tag)
        n_clusts_new = new_data[tag].nunique()
        reserve = new_data.copy()
        reserve['ind'] = new_data.index
        
        while n_clusts_new<n_clusts:
            reserve_cols = [x for x in self.cluster_cols if x not in keep+[old_tag]]
            new_cluster_vars = [tag] + keep
            new_index_vars = [x for x in self.index_cols if x not in 
                reserve_cols+new_cluster_vars+[old_tag]]
    
            reserve['ind'] = reserve[tag]
            new_data = new_data.groupby(new_index_vars+new_cluster_vars)
            new_data = new_data[self.value_var].sum().reset_index()
    
            self.__init__(data=new_data, value_var=self.value_var, 
                cluster_vars=new_cluster_vars, index_vars=new_index_vars)
            self.corr_dist(method=corr_method, weight=weight, absolute=absolute,
                           auto_exclude=auto_exclude, scale=scale, 
                           transform=transform)
            self.linkage(method=linkage_method, **linkage_kwargs)
            self.assign_clusters(t=t, criterion=criterion, **fclust_kwargs)
    
            i += 1
            old_tag = tag
            tag = 'cluster_id_'+str(i)
            new_data = self.append_cluster_col(tag=tag)
            n_clusts = n_clusts_new
            n_clusts_new = new_data[tag].nunique()
            merge_data = new_data[[old_tag, tag]].drop_duplicates().set_index(old_tag)
            reserve[tag] = reserve['ind'].map(merge_data.squeeze().to_dict())
            
            new_data = new_data.groupby(new_index_vars+keep+[tag])
            new_data = new_data[self.value_var].sum().reset_index()            
            
            self.__init__(data=new_data, value_var=self.value_var, 
                cluster_vars=keep+[tag], index_vars=new_index_vars)
            
            self.aggregated_data = new_data
            self.recursive_data = reserve.drop(['ind'], axis=1)
        
    def summarize(self):
        self.collect_groups()

        if type(self.correlation_matrix.index)==pd.core.index.MultiIndex:
            levels = self.correlation_matrix.index.levels
            levels = [np.unique(x.values) for x in levels]
            levels = [len(x) for x in levels]
            multiple = len(set(levels))>1
        else:
            multiple = False

        if multiple:
            self.summarize_multiindex()
        else:
            self.summarize_singleindex()
            
        return self.summary
        
    def quick_run(self, corr_method='pearson', weight=True, auto_exclude=None,
        absolute=False, link_method='complete', threshold=0.0525, scale=False,
        criterion='distance', transform=None, plot=True):
        self.corr_dist(method=corr_method, weight=weight, absolute=absolute,
                       auto_exclude=auto_exclude, scale=scale, 
                       transform=transform)
        self.linkage(method=link_method)
        self.assign_clusters(t=threshold, criterion=criterion)
        self.dendrogram(plot=plot)
        self.summarize()
        return self.summary

class Ensemble(object):
    def __init__(self, data, yname, prep=True):
        self.data_ = data.copy()
        self.prep = prep
        self.yname = yname
        self.scaler = {'mean':0, 'std':1}

        if prep:
            move = [x not in [np.float, np.int, np.bool] for x in data.dtypes]
            ind = data.dtypes.index[move]
            if len(ind)>0:
                self.data_ = data.copy().set_index(ind).astype('float64')
                
    def _save_state(self):
        saver = dict(data=self.data_.copy(), yname=self.yname, 
             prep=self.prep, modeler = clone(self.modeler))
        self.saved_state_  = dict(saver)
    
    def _load_state(self):
        old_data = self.saved_state_['data']
        old_yname = self.saved_state_['yname']
        old_prep = self.saved_state_['prep']
        old_modeler = self.saved_state_['modeler']

        self.__init__(old_data, yname=old_yname, prep=old_prep)
        self.load_modeler(old_modeler)

    def _scale_data(self, input_data):
        scaled = input_data.copy()
        scaled = ((scaled - self.scaler['mean']) / self.scaler['std'])
        scaled = scaled.fillna(0)
        return scaled

    def load_modeler(self, obj):
        self.modeler = clone(obj)

    @staticmethod
    def bootstrap_pandas(df, nboot, method='arithmetic'):
        if type(df)==pd.Series:
            df = df.copy().to_frame()
        if method=='arithmetic':
            method = np.mean
        elif method=='geometric':
            method = geometric_mean
        n = df.shape[0]
        groups = np.repeat(np.arange(nboot), n)
        boot = np.random.choice(n, n*nboot)
        out = df.iloc[boot,:].groupby(groups).apply(method)
        return out.squeeze()

    def fit(self, scale=True, seed=None):
    
        if scale:
            self.scaler = {'mean': self.data_.mean(), 'std': self.data_.std()}

        self.training_data_scaled_ = self._scale_data(self.data_)
        new_data = self.training_data_scaled_.copy()
        
        x_data = new_data.drop([self.yname], axis=1)
        y_data = new_data[self.yname]
        
        if seed is not None:
            np.random.seed(seed)
        _ = self.modeler.fit(X=x_data, y=y_data)

        imps = self.modeler.feature_importances_
        self.feature_importances_ = pd.Series(imps, index=x_data.columns)
        self.feature_importances_.sort(ascending=False)

    def predict(self, test_data):
    
        self.test_data_ = test_data.copy()    
        new_data = self._scale_data(test_data)
        self.test_data_scaled_ = new_data.copy()
            
        for_prediction = new_data.drop([self.yname], axis=1)
        preds = pd.Series(index=for_prediction.index)
        preds.loc[:] = self.modeler.predict(for_prediction)
        if all([type(x)==pd.Series for x in self.scaler.values()]):            
            preds = (preds * self.scaler['std'][self.yname]) + \
                self.scaler['mean'][self.yname]
            
        actuals = self.test_data_[[self.yname]].copy()
        model_preds = pd.DataFrame(index=preds.index)
        model_preds['predicted'] = preds.astype('float64')
        model_preds['actual'] = actuals.astype('float64')

        self.predictions_ = model_preds
            
    def simulate(self, predictors, n_inc=100, n_compare=100, multiply=None,
            var_level='variable', inc_type='actual'):
        
        sim_data = self.training_data_scaled_.copy()        

        if multiply is None:
            mult = int(np.log(len(predictors)))+1
        else:
            mult = multiply
        idx = pd.MultiIndex.from_tuples(predictors, 
                                        names=self.data_.columns.names)
        sim_vals = pd.DataFrame(columns=idx)
        sim_vals_raw = pd.DataFrame(columns=idx)
        for p in predictors:
            if inc_type=='actual':
                sims = np.linspace(sim_data[p].min(), sim_data[p].max(), 
                                   n_inc).tolist()
                sims_raw = np.linspace(self.data_[p].min(), self.data_[p].max(), 
                                   n_inc).tolist()
            elif inc_type=='quantile':
                sims = [sim_data[p].quantile(x) for x in 
                    np.linspace(0, 1, n_inc)]
                sims_raw = [self.data_[p].quantile(x) for x in 
                    np.linspace(0, 1, n_inc)]
            elif inc_type=='random':
                sims = np.random.choice(sim_data[p], n_inc).tolist()
                sims_raw = np.random.choice(self.data_[p], n_inc).tolist()

            sims_ind = np.random.choice(n_inc, n_inc*n_compare*mult)
            sim_vals[p] = np.array(sims)[sims_ind]
            sim_vals_raw[p] = np.array(sims_raw)[sims_ind]
        
        inds = np.random.choice(sim_data.shape[0], n_inc*n_compare*mult)
        
        new_data = sim_data.iloc[inds, :].copy()
        new_data.loc[:,predictors] = sim_vals.values
        for_prediction = new_data.drop([self.yname], axis=1)        
        
        preds = pd.Series(index=for_prediction.index)
        preds.loc[:] = self.modeler.predict(for_prediction)
        if all([type(x)==pd.Series for x in self.scaler.values()]):            
            preds = (preds * self.scaler['std'][self.yname]) + \
                self.scaler['mean'][self.yname]

        output = sim_vals_raw[predictors] 
        preds = preds.reset_index(drop=True).to_frame('predicted')
        output = output.reset_index(drop=True)
        
        if type(output.columns) == pd.MultiIndex:
            stack_levels = [x for x in output.columns.names if x!=var_level]
            output = output.stack(stack_levels).reset_index(stack_levels)
        else:
            stack_levels = []
        
        output = pd.merge(output, preds, how='outer', left_index=True, 
                          right_index=True)
        output = output.reset_index(drop=True)
        output.columns.names = ['variable']
        output = output.set_index(stack_levels+['predicted'])
        output = output.stack().to_frame('value')
        output = output.reset_index()
    
        return output
    
    def crossvalidate(self, fold_index, scale=True, seed=None):
        self._save_state()
        train = self.data_.loc[fold_index,:].copy()
        test = self.data_.loc[~fold_index,:].copy()
        self.data_ = train
        self.fit(scale=scale, seed=seed)
        self.predict(test)
        self._load_state()
        return copy.deepcopy(self)

    def kfold(self, nfolds=10, **kwargs):

        self.kfold_results_ = None

        n = self.data_.shape[0]
        reps = (n//nfolds)+1
        folds = np.repeat(range(nfolds), reps)[:n]
        folds = np.random.permutation(folds)
        fold_list = [self.crossvalidate(folds!=fold, **kwargs) for fold in 
            np.unique(folds)]
    
        self.kfold_results_ = fold_list

    def model_performance(self, verbose=True):
        
        preds = pd.concat([x.predictions_ for x in self.kfold_results_])
        self.kfold_predictions_ = preds.copy()

        if verbose:
            preds = preds[['actual', 'predicted']]            
            kfold_preds = pd.DataFrame(columns=['pearson', 'kendall'], index=[0])
            kfold_preds['pearson'] = preds.corr().iloc[0,1]
            kfold_preds['kendall'] = preds.corr(method='kendall').iloc[0,1]
            kfold_preds['amean_abs'] = preds.T.diff().abs().T.iloc[:,1].mean()
            kfold_preds['gmean_abs'] = np.exp(np.log(preds.T.diff().abs(
                ).T.iloc[:,1]+1).mean())-1
            kfold_preds['median_abs'] = preds.T.diff().abs(
                ).T.iloc[:,1].median()
            kfold_preds['amean_pct'] = preds.T.pct_change().abs(
                ).T.iloc[:,1].replace({np.inf:np.nan, -np.inf:np.nan}).mean()
            kfold_preds['gmean_pct'] = np.exp(np.log(preds.T.pct_change().abs(
                ).T.iloc[:,1].replace({np.inf:np.nan, -np.inf:np.nan})+1).mean()
                )-1
            kfold_preds['median_pct'] = preds.T.pct_change().abs(
                ).T.iloc[:,1].replace({np.inf:np.nan, -np.inf:np.nan}).median()
            slope = LinearRegression().fit(
                X=preds[['predicted']], y=preds['actual']).coef_[0]
            kfold_preds['slope'] = slope

            return kfold_preds

    def _get_tree_importances(self, models, combine=False):
    
        try:
            outputs = [est.feature_importances_ for hold in 
                models.estimators_ for est in hold]
        except:
            outputs = [est.feature_importances_ for est in 
                models.estimators_]

        out = pd.DataFrame(np.column_stack(outputs).T, 
                columns=self.data_.drop([self.yname], axis=1).columns)        

        return out
        
    def select_features(self, rimp_threshold=0.01, imp_threshold=None, 
                        droplevel=None):

        self._save_state()
        features = self.feature_evaluations_.copy()
        keep = np.ones(features.shape[0], dtype=np.bool)
        if rimp_threshold is not None:
            keep[features['relative_importance'].values<rimp_threshold] = False
        if imp_threshold is not None:
            keep[features['mean_importance'].values<imp_threshold] = False
        
        keepers = features.index[keep]
        
        if droplevel is not None:
            original_items = features.index
            all_items = original_items            

            for l in droplevel:
                all_items = all_items.droplevel(l)
                keepers = keepers.droplevel(l)
            keep = np.array([True if x in keepers else False for x in all_items])
            keepers = features.index[keep]
                    
        new_data = self.data_.loc[:,keepers].copy()
        new_data[self.yname] = self.data_[self.yname].copy()
        self.load_state()

    @staticmethod
    def quantile_lift(data, n=10, metric=sum, plot=False):
        original = data['actual'].copy()
        predicted = data['predicted'].copy()
        q = 1.0/n
        qs = np.arange(0.0, 1.0+q, q)
        levels = [original.quantile(x) for x in qs[1:-1]]
        levels = [-np.inf] + levels + [np.inf]
        grouper = pd.cut(original, bins=levels, labels=qs[1:].astype(str))
        output= pd.DataFrame()
        output['actual'] = original.groupby(grouper).apply(metric)
        output['predicted'] = predicted.groupby(grouper).apply(metric)
        if plot:
            output.plot()
        return output
    

class ProductSpace(object):
    def __init__(self, data, products, geographies, value_var, year_var=None):
        self.original_data = data.copy().dropna(subset=[value_var])
        self.product_vars = products[:]
        self.geography_vars = geographies[:]
        self.value_var = value_var
        self.year_var = year_var
        if year_var is not None:
            self.years = data[year_var].unique().tolist()
        else:
            self.years = None

    @staticmethod
    def lowess_diff(data, target, by, delta_percent=0.01, **kwargs):
        x = data[by].copy()
        y = data[target].copy()
        d = (y.max() - y.min()) * delta_percent
        if np.isnan(d):
            d=0.0
        lowess_line = smoother.lowess(endog=x, exog=y, delta=d, 
                                      return_sorted=False, **kwargs)
        output = x - lowess_line
        return output
        
    def reshape(self, method='division', cutoff='auto', **kwargs):
        '''
        method: 'division' or 'lowess'
        cutoff: threshold for binarizing reults of transforming self.value_var
        by method
        kwargs: passed to smoother.lowess
        '''
        
        data = self.original_data.copy()
        share_vars = self.geography_vars[:]
        global_share_vars = self.product_vars[:]
        if self.years is not None:
            share_vars += [self.year_var]
            global_share_vars += [self.year_var]

        if method=='division':
            if cutoff=='auto':
                cutoff = 1
                
            share_index = data[share_vars].to_records(index=False)
            share = data.groupby(share_vars, sort=False)[self.value_var]
            share = share.apply(lambda x: x/x.sum())
            
            global_share_index = data[global_share_vars].to_records(index=False)
            global_share = data.groupby(global_share_index, sort=False)
            global_share = global_share[self.value_var]
            global_share = global_share.transform(np.sum)
            global_share = global_share.groupby(share_index, sort=False).apply(
                lambda x: x/x.sum())
                
            scale = share / global_share

        elif method=='lowess':
            if cutoff=='auto':
                cutoff = 0

            def lowess_wrapper(x):
                out = self.lowess_diff(data=x, target=self.value_var, **kwargs)
                return out
                
            scale = data.groupby(global_share_vars, sort=False, squeeze=True)
            scale = scale.apply(lowess_wrapper)
        
        if cutoff is not None:
            data['decision'] = (scale.values > cutoff) * 1
        else:
            data['decision'] = scale
        
        axis_cols = self.product_vars + self.geography_vars

        if self.year_var is not None:
            axis_cols += [self.year_var]

        data = data.set_index(axis_cols)['decision']
        data = data.unstack(self.geography_vars).fillna(0)
        
        self.data = data

    def proximity(self):
        data = self.data.copy()
            
        n = data.sum(axis=1)
        proximity = data.dot(data.T) / n
        prox_u = np.triu(proximity)
        prox_l = np.tril(proximity)
        prox_new = np.where(prox_u < prox_l.T, prox_u, prox_l.T)
        proximity = pd.DataFrame(prox_new + prox_new.T, index=proximity.index,
            columns=proximity.columns)
        proximity.values[np.diag_indices_from(proximity)] = 1

        if self.years is not None:
            c_y = proximity.columns.get_level_values(self.year_var)
            i_y = proximity.index.get_level_values(self.year_var)
            mask = [i!=c_y.values for i in i_y.values]
            mask = np.column_stack(mask)
            proximity.values[mask] = np.nan
            proximity.columns = proximity.columns.droplevel(self.year_var)
            proximity.columns.names = [x+'_2' for x in self.product_vars]

        proximity = proximity.stack().to_frame('proximity').reset_index()

        return proximity
        
    def density(self, proximity):
        index_cols = [x for x in proximity.columns if x!='proximity']
        unstack_cols = [x+'_2' for x in self.product_vars]
        prox = proximity.copy().set_index(index_cols)['proximity']
        prox = prox.unstack(unstack_cols)
        prox.columns.names = [x.rstrip('_2') for x in prox.columns.names]
        
        data = self.data.copy()
        
        density = pd.DataFrame(columns=data.columns, index=prox.index)
        for col in data.columns:
            density[col] = (prox * data[col]).sum(axis=1) / prox.sum(axis=1)
        density = density.stack().to_frame('density').reset_index()
        
        return density
                
    def calculate(self, verbose=True, **kwargs):
        if verbose:
            print 'reshaping data'
        self.reshape(**kwargs)
        n = self.data.sum(axis=1).to_frame('n').reset_index()
        if verbose:
            print 'calculating proximities'
        proximity = self.proximity()
        if verbose:
            print 'calculating densities'
        density = self.density(proximity)
        
        return n, proximity, density

def empirical_error(test, train_predict, train_actual, n=1000, q=None, 
                    lower_bound=None, **kwargs):

    # calculate difference between predicted and actual
    diffs = train_actual.astype(float) - train_predict.astype(float)
    if type(diffs)==pd.Series:
        diffs = diffs.values
    if type(train_actual)==pd.Series:
        actuals = train_actual.values
    else:
        actuals = train_actual

    error_list = []
    for t in test:
        # calculate absolute ratio of test to actuals
        shift = 0.0
        if t==0.0:
            ratio = np.abs((t+1.0)/(actuals+1.0))            
        else:            
            ratio = np.abs((t+shift)/(actuals+shift))
            shift = shift + 1
        ratio[np.isnan(ratio)] = np.abs((t+shift)/(actuals+shift))[np.isnan(ratio)]
        ratio[np.isinf(ratio)] = np.abs((t+shift)/(actuals+shift))[np.isinf(ratio)]
        
        #scale actual differences by ratio, then add to test (optional cutoff)
        scaled_diffs = diffs*ratio
        candidate = t + scaled_diffs
        if lower_bound is not None:
            candidate[candidate<lower_bound] = lower_bound

        # randomly select n errors, then sort
        errors = np.random.choice(candidate, n)
        errors.sort()
        error_list.append(errors)
    
    errors = np.column_stack(error_list)
    try:
        idx = test.index
    except:
        idx = range(errors.shape[1])
    errors = pd.DataFrame(errors, columns=idx)
    output = pd.DataFrame({'estimate':test})
    if q is None:
        output = pd.concat([output, errors.T], axis=1)
        output = output.set_index(['estimate'], append=True)
        output.columns.names = ['simulation']
        output = output.stack().to_frame('error').reset_index(
            ['estimate', 'simulation'])
    else:
        for prob in q:
            output['q_'+str(errors)] = errors.quantile(errors)
    
    return output

def corr_data(data, i, value_var, time_var, index_vars, min_time=None, t=None,
              weight=False, fill=True, return_df=True):
    t0 = time.time()
    print 'shaping data sets...'
    
    if min_time is None:
        min_t = data[time_var].max()
    else:
        min_t = min_time
        
    df = data.set_index(index_vars+[time_var])[value_var].unstack(index_vars)
    output_df = df.T.stack().to_frame(value_var).reset_index()
    output_df = output_df.set_index(index_vars+[time_var])
    output_df = output_df[value_var].unstack(index_vars)
    
    df0 = df.T.stack().to_frame(value_var).reset_index()
    df0 = df0[df0[time_var]>=min_t]
    df0 = df0.set_index(index_vars+[time_var])
    df0 = df0[value_var].unstack(index_vars)

    dfi = df.shift(i).T.stack().to_frame(value_var).reset_index()
    dfi = dfi[dfi[time_var]>=min_t]
    dfi = dfi.set_index(index_vars+[time_var])
    dfi = dfi[value_var].unstack(index_vars)
    
    if fill:
        df0 = df0.fillna(0)
        dfi = dfi.fillna(0)
        
    col_ind = len(df0.columns)
    df = pd.concat([df0, dfi], axis=1)

    print 'calculating correlations...'
    corrs = df.corr('pearson', min_periods=i)
    if weight:
        jaccard = pairwise_distances(df.T.notnull().values, metric='jaccard',
                               n_jobs=-1)
        jaccard = pd.DataFrame(jaccard, columns=corrs.columns, index=corrs.index)
    
    print 'filtering correlations...'
    corrs = corrs.iloc[:col_ind,(col_ind+1):]
    corrs = corrs.fillna(0)
    if weight:
        jaccard = jaccard.iloc[:col_ind,(col_ind+1):]
        jaccard = 1 - jaccard
        corrs = corrs * jaccard
    column_names = [str(x)+'_compare' for x in corrs.columns.names]
    corrs.columns.names = column_names
    
    if t is not None:        
        corrs = corrs.stack(column_names)
        corrs = corrs[corrs>=t].sort_index()
        corrs = corrs.unstack(column_names)

    print 'total time:', time.time() - t0
    if return_df:
        return output_df, corrs
    else:
        return corrs

from sklearn.decomposition import KernelPCA
from sklearn.cluster import AffinityPropagation
import matplotlib.pyplot as plt

def get_continuity_start(arr, rounder=None):
    d = arr.values
    d = np.diff(d)
    if rounder is not None:
        d = np.round(d, rounder)
    out = []
    j = 0
    for i in range(1, len(d)):
        if d[i-1]!=d[i]:
            j = i
        out.append(j)
    
    s = pd.Series(out).value_counts()
    if s.iloc[0]==1:
        print 'no good choice'
    else:
        return arr.index[s.index[0]]

def percent_majority(x):
    signed = np.sign(x)
    top = np.sign(x).astype(str).describe().T['top'].astype(float)
    percent = (signed!=top).sum()
    return percent

def affinity_propagation(df, weights=None, rounder=3, output='object', 
                         verbose=False):
    if type(df)==pd.DataFrame:
        df1 = df.values
    else:
        df1 = df
    dists = pairwise_distances(df1).flatten()**2
    dists = dists[dists>0.0]
    inputs = np.linspace(-np.median(dists)/2, -dists.min(), 101)[:-1]
    ds = np.arange(0.5, 1.0-0.01, .01)

    out = []
    if verbose:
        print 'exploring damping options...'
    for d in ds:
        if verbose:
            print d, 
        km = AffinityPropagation(preference=inputs[0], damping=d, 
                                 convergence_iter=30, max_iter=400)
        x = km.fit(df1) 
        labels = pd.Series(km.labels_).value_counts(normalize=True).sort_index()
        maxs = df.groupby(km.labels_).max() 
        mins = df.groupby(km.labels_).min()
        rang = maxs - mins
        rang = (rang.T * labels).T.sum() / labels.sum()
        if weights is not None:
            rang = (rang * np.array(weights)).sum() / np.array(weights).sum()
        else:
            rang = rang.mean()
        out.append(rang)
    
    out = pd.Series(out, index=ds)
    damp = get_continuity_start(out, rounder)
    damps = out.copy()
    
    out = []
    counts = []
    n = []
    if verbose:
        print ''
        print 'exploring preference options...'
    for i in inputs:
        if verbose:
            print np.round(i, 2),
        km = AffinityPropagation(preference=i, damping=damp, 
                                 convergence_iter=30, max_iter=400)
        x = km.fit(df1) 
        labels = pd.Series(km.labels_).value_counts(normalize=True).sort_index()
        maxs = df.groupby(km.labels_).max() 
        mins = df.groupby(km.labels_).min()
        rang = maxs - mins
        rang = (rang.T * labels).T.sum() / labels.sum()
        if weights is not None:
            rang = (rang * np.array(weights)).sum() / np.array(weights).sum()
        else:
            rang = rang.mean()

        new_labels = labels.copy()
        new_labels = 1.0 - new_labels
        new_labels[pd.Series(km.labels_).value_counts().sort_index()==1] = 0.0
        percents = df.groupby(km.labels_).apply(percent_majority)
        #mask = percents.sum(axis=1)!=1.0
        #percents = (percents[mask].T * new_labels[mask]).T.sum() / new_labels[mask].sum()
        percents = percents.sum()
        if weights is not None:
            percents = (percents * np.array(weights)).sum() / np.array(weights).sum()
        else:
            percents = percents.mean()

        #percents = percents.values.sum()

        out.append(rang)
        counts.append(percents)
        n.append(labels.shape[0])

    out = pd.DataFrame({'mean_range':out, 'total_count':counts, 'n_groups':n}, 
                       index=inputs)
    
    min_ind = get_continuity_start(out['n_groups'].pct_change(),rounder)
    max_ind = (out.loc[min_ind:,'n_groups'].pct_change().dropna().abs().round(rounder)!=0.0).argmax()
    pref = out.loc[min_ind:max_ind,:].iloc[:-1,:]
    if len(pref)==0:
        print ''
        print 'mean_corr and n_mult criteria exclude all possibilities'
        return out
    else: 
        pref = pref[pref['total_count']==pref['total_count'].min()].index[0]
    
    if output=='object':
        km = AffinityPropagation(preference=pref, damping=damp, 
                                 convergence_iter=30, max_iter=400)
        x = km.fit(df1) 
        return km
    else:
        return damps, out, damp, pref

def principal_components(data, idx, val, kernel='linear', cluster=True,
                         components=None, sensitivity=3, verbose=False,
                         output_dict=None):
    try:
        df = data[val].unstack(idx).fillna(0).T
    except:
        df = data.set_index(idx)[val].unstack().fillna(0).T
    df = np.log(df+1)
    df = (df - df.mean()) / df.std()
    df = df.T.corr()
    sp = KernelPCA(n_components=min(*df.shape), kernel=kernel)
    proj = sp.fit_transform(df)
    var_expl = sp.lambdas_
    var_expl[var_expl<0.0] = 0.0
    var_expl = var_expl/var_expl.sum()
    proj = pd.DataFrame(proj, index=df.index)
    proj.columns = [x+1 for x in proj.columns]
    loadings= pd.concat([df, proj], axis=1).corr()
    loadings = loadings.loc[:, proj.columns].loc[df.columns,:]
    keep = (var_expl>(2.0/df.shape[1])).sum()
    loadings = loadings.iloc[:,:keep]    
    variance_explained = pd.DataFrame(index=loadings.columns)
    variance_explained['total'] = var_expl[:keep]
    variance_explained['cumulative'] = var_expl.cumsum()[:keep]
    variance_explained['max_abs_corr'] = loadings.abs().max()

    if components is None:
        load_cols = variance_explained.index
    else:
        load_cols = components

    output = loadings.copy()
    
    if cluster:
        ap = affinity_propagation(df=loadings[load_cols], 
            weights=variance_explained['total'][load_cols].values, 
            rounder=sensitivity, verbose=verbose)
        output['cluster'] = ap.labels_

        if output_dict is not None:
            level_values = output.index.get_level_values(output_dict).values
            cat_dict = pd.Series(level_values).groupby(ap.labels_).unique()
            cat_dict = cat_dict.to_dict()
            cat_dict = {k:v.tolist() for k,v in cat_dict.items()}
        
    class final(object):
        correlation = df.copy()
        diagnostic = variance_explained
        pca_object = sp
        ap_object = ap
        loading = output
        cluster_dictionary = cat_dict
    
    final_output = final()
    
    return final_output
        
def plot_components(loadings, labels=None, components=None):
    loads = loadings.copy()
    if labels is not None:
        loads['cluster'] = labels

    if components is not None:
        x_lab = components[0]
        y_lab = components[1]
    else:
        x_lab = 1
        y_lab = 2

    if labels is None:
        plt.plot(loadings[x_lab], loadings[y_lab], '.')
    else:
        plt.figure()

    x = plt.xlim(-1.0, 1.0)
    x = plt.ylim(-1.0, 1.0)
    x = plt.yticks(np.linspace(-1.0, 1.0, 9), fontsize=7)
    x = plt.xticks(np.linspace(-1.0, 1.0, 9), fontsize=7)
    x = plt.axhline(y=0, xmin=-1.0, xmax=1.0)
    x = plt.axvline(x=0, ymin=-1.0, ymax=1.0)
    
    if labels is not None:    
        for i in loads.index:
            x = plt.text(loads.loc[i, x_lab], loads.loc[i, y_lab], 
                str(int(loads.loc[i,'cluster'])), fontdict={'size': 6})
    plt.show()

def plot_correlations(corr, loadings, component=1, cmap='RdBu',
                          tick_level=None):
    df_corr = corr.copy()
    if type(component)==int:
        loads = loadings[[component]].mean(axis=1).order().copy()
        col_order = loads.index
    elif component=='naive':
        col_order = df_corr.sum().order().index
    df_plot = df_corr.loc[col_order, col_order].copy()

    x = plt.pcolor(df_plot, cmap=cmap, vmin=-1.0, vmax=1.0)
    x = plt.xlim(0, df_plot.shape[0])
    x = plt.ylim(0, df_plot.shape[1])
    if tick_level is not None:
        x = plt.yticks(np.arange(0.5, df_plot.shape[0], 1), 
                       df_plot.index.get_level_values('emcategory'), 
                       fontsize=7)
        x = plt.xticks(np.arange(0.5, df_plot.shape[1], 1), 
                       df_plot.columns.get_level_values('emcategory'), 
                       rotation=270, fontsize=7)
    else:
        x = plt.yticks(np.arange(0.5, df_plot.shape[0], 1), fontsize=7)
        x = plt.xticks(np.arange(0.5, df_plot.shape[1], 1), rotation=270, 
            fontsize=7)

    plt.show()
