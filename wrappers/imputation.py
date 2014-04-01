# -*- coding: utf-8 -*-
"""
Created on Mon Jan 27 13:15:17 2014
@author: schaunwheeler
"""

import sklearn.ensemble as sk_en
import numpy as np
import pandas as pd
import multiprocessing
import wrappers




def miss_forest(xmis, maxiter=10, n_estimators=100, verbose=False,
    max_features='sqrt', bootstrap=True, nodesize=None, max_depth=None,
    min_samples_split=2, cores='auto', random_state=None, cols_to_dummies=None,
    keep_imputations=False, leave_cols=None, ignore_cols=None, 
    grouping_variables=None, summary_covariates=None, summary_groups=None,
    error_threshold=None, lag_cols=None, lead_cols=None, n_shift=1):
    '''
    Partial port of R MissForest package for nonparametric missing value
    imputation (original author: D.Stekhoven, stekhoven@stat.math.ethz.ch)

    Parameters
    ----------
    xmis : data matrix with missing values
    maxiter : stop after how many iterations (default = 10)
    n_estimators : how many trees are grown in the forest (default = 100)
    verbose : (boolean) if TRUE then missForest returns error estimates,
        runtime and if available true error during iterations
    max_features : how many variables should be tried randomly at each node
    bootstrap : (boolean) if TRUE bootstrap sampling (with replacement)
        is performed
    min_samples_split : minimum size of terminal nodes, vector of length 2, with
        number for continuous variables in the first entry and number for
        categorical variables in the second entry
    max_depth : maximum number of terminal nodes for individual trees
    cores : number of cores among which to distribute random forest compuation
    cols_to_dummies : columns of xmis to turn into dummy variables before
        imputation
    keep_imputations : keep a complete list of the entire data set at each
        iteration of the imputation process
    ignore_cols : columns for which missing values should not be imputed; these
        will still have missing values filled in with column means and be
        included in the imputation of missing values for other columns; columns
        with no missing values do not need to be included in this list
    grouping_variables : columns names of variables to use as groups when
        calcluating averages and mean absolute scaled error

    Returns
    -------
    A class containing the following:

    random_forest : the scikit-learn random forest regressor class from the
        last iteration of the imputation
    oob_error = a pandas data frame containing the out-of-bag prediction error
        (as measured by mean absolute scaled error) for each column and each
        iteration
    imputation_difference = the sum of squares difference between the final
        imputation and the one previous
    kept_imputations = (if keep_imputations=True), a list of imputed data sets
        from all iterations
    output_data = the imputed data set from the final iteration


    '''

    xmis = xmis.replace([-np.inf, np.inf], np.nan)


    if grouping_variables is not None:
        grouping_index = xmis.set_index(grouping_variables).index

    if summary_groups is not None:
        summary_index = xmis.set_index(summary_groups).index

    if cols_to_dummies is not None:
        if verbose > 0:
            print 'creating dummy variables'
        dummies = []
        cols_from_dummies = xmis[cols_to_dummies].copy()
        for col in cols_to_dummies:
            dummies.append(pd.get_dummies(xmis[col], prefix=col,
                prefix_sep='___', dummy_na=True))
        xmis = pd.concat([xmis] + dummies, axis=1)
        xmis = xmis.drop(cols_to_dummies, axis=1)
        dummies = dict(zip(cols_to_dummies,
            [list(x.columns.values) for x in dummies]))

    if ignore_cols is not None:
        ignored_df = xmis[ignore_cols]
        xmis = xmis.drop(ignore_cols, axis=1)

    if(cores == "auto"):
        if(multiprocessing.cpu_count() > 1):
            cores = multiprocessing.cpu_count() - 1
        else:
            cores = 1

    ## extract missingness pattern
    if verbose > 0:
        print 'identifying missingness patterns'
    na_locations = xmis.isnull()
    percent_missing = na_locations.mean()
    percent_missing.sort()
    xmis = xmis[percent_missing.index].copy()
    cols_to_impute = (percent_missing > 0) & (percent_missing < 1.0)
    cols_to_impute = percent_missing[cols_to_impute].index.values
    cols_to_impute = pd.Series(cols_to_impute)

    if leave_cols is not None:
        cols_to_impute = cols_to_impute[~cols_to_impute.isin(leave_cols)]
    
    ## remove completely missing variables
    if (percent_missing==1.0).sum() > 0:
        cols_removed = percent_missing[percent_missing == 1.0].index.values
        print 'removed variable(s) %s due to the missingness of all entries' % (
              ', '.join(cols_removed))
    if verbose > 0:
        print 'the following columns will be imputed: %s' % (
            ', '.join(cols_to_impute))
 
    ## perform initial S.W.A.G. on xmis (mean imputation)
    if verbose > 0:
        print 'filling in missing values with mean'
        
    keep_cols = percent_missing.index.values[(percent_missing != 1.0).values]
    ximp = xmis[keep_cols].copy()

    if grouping_variables is not None:
        ximp = ximp.groupby(grouping_index, sort=False)
        ximp = ximp.fillna(ximp.mean())
    else:
        ximp = ximp.fillna(ximp.mean()).copy()

    if lead_cols is not None:
        if grouping_variables is not None:
            x_lead = ximp[lead_cols].groupby(grouping_index, sort=False)
            x_lead = x_lead.shift(-n_shift)
        else:
            x_lead = ximp[lead_cols].shift(-n_shift)
        
        x_lead = x_lead.fillna(0)                
        x_lead.columns = [x+'__lead' for x in x_lead.columns]
        ximp = pd.concat([ximp, x_lead], axis=1)

    if lag_cols is not None:
        if grouping_variables is not None:
            x_lag = ximp[lag_cols].groupby(grouping_index, sort=False)
            x_lag = x_lag.shift(n_shift)
        else:
            x_lag = ximp[lag_cols].shift(n_shift)
        
        x_lag = x_lag.fillna(0)                
        x_lag.columns = [x+'__lag' for x in x_lag.columns]
        ximp = pd.concat([ximp, x_lag], axis=1)

    ## initialize parameters of interest
    iteration = 1
    conv_new = 0
    conv_old = np.inf
    oob_error = pd.DataFrame(index=cols_to_impute)
    all_cols = [x for x in keep_cols]

    if summary_covariates is not None:    
        all_cols = all_cols + summary_covariates.keys()    
    if lead_cols is not None:    
        all_cols = all_cols + [x+'__lead' for x in lead_cols]
    if lag_cols is not None:    
        all_cols = all_cols + [x+'__lag' for x in lag_cols]
    
    importance_scores = pd.DataFrame(index=all_cols)

    ## function to yield the stopping criterion in the following 'while' loop
    def stop_criterion(conv_new, conv_old, iteration, maxiter):
      output = (conv_new < conv_old) & (iteration <= maxiter)
      return output

    x_imputations = []
    ## iterate missForest
    while stop_criterion(conv_new, conv_old, iteration, maxiter):
        if iteration != 1:
            conv_old = conv_new

        iteration_name = 'iteration_'+str(iteration)
        iteration += 1
        oob_error[iteration_name] = np.nan
        ximp_old = ximp.copy()

        if verbose > 0:
            print iteration_name.replace('_', ' ')
            print 'Columns (out of %d) imputed:' % len(cols_to_impute)

        for col in cols_to_impute:
            if verbose > 0:
                print (list(cols_to_impute).index(col) + 1),

            if summary_covariates is not None:
                index_names = summary_index.names
                add_index = pd.DataFrame(summary_index.tolist(), 
                    columns=index_names)
                ximp = pd.concat([ximp, add_index], axis=1)
                ximp = wrappers.convenience.summary_covariates(ximp, 
                    cols=[col], group_cols=summary_groups, sep_string='___', 
                    funcs=summary_covariates)
                ximp = ximp.drop(index_names, axis=1)

            observed = ~na_locations[col]
            x_cols = [x for x in ximp.columns if x != col]            

            obs_y = ximp.ix[observed, col].squeeze().copy()
            mis_y = ximp.ix[~observed, col].squeeze().copy()
            
            obs_x = ximp.ix[observed, x_cols].copy()
            mis_x = ximp.ix[~observed, x_cols].copy()

            rfr = sk_en.RandomForestRegressor(n_estimators=n_estimators,
                max_features=max_features, oob_score=True, verbose = False,
                n_jobs = cores,random_state=random_state,
                min_samples_split=min_samples_split,
                bootstrap=bootstrap, max_depth=max_depth)

            _ = rfr.fit(X = obs_x, y = obs_y)

            # record out-of-bag error
            if grouping_variables is not None:
                grp_index = grouping_index[observed]
                oob_error_val = wrappers.convenience.mase(obs_y.values,rfr.oob_prediction_,
                    grouping=grp_index.values)
                oob_error.loc[oob_error.index==col, iteration_name] = oob_error_val
            else:
                oob_error_val = wrappers.convenience.mase(obs_y.values,rfr.oob_prediction_)
                oob_error.ix[oob_error.index==col, iteration_name] = oob_error_val
            
            imp_scores = pd.Series(rfr.feature_importances_, index=obs_x.columns)
            rename_dict = dict(zip(
                [col+'___'+x for x in summary_covariates.keys()], 
                     summary_covariates.keys()))
            imp_scores = imp_scores.rename(index=rename_dict)
            importance_scores[col] = imp_scores

            # predict missing parts of Y
            mis_y = rfr.predict(mis_x)
            ximp.ix[~observed, col] = mis_y
            
            if summary_covariates is not None:
                drop_cols = [col+'___'+x for x in summary_covariates.keys()]

        if lead_cols is not None:
            drop_cols = [x+'__lead' for x in lead_cols]
            ximp = ximp.drop(drop_cols, axis=1)

        if lag_cols is not None:
            drop_cols = [x+'__lag' for x in lag_cols]
            ximp = ximp.drop(drop_cols, axis=1)
                
        if keep_imputations:
            x_imputations.append(ximp)

        ## check the difference between iteration steps
        conv_new = ((ximp[cols_to_impute] - ximp_old[cols_to_impute])**2 /
            (ximp[cols_to_impute]**2).sum()).sum().sum()

        ## return status output, if desired
        oob_means = oob_error[iteration_name].mean()
        percent_majority = (oob_error[iteration_name] > 0.5).mean()
        if verbose > 0:
            print ''
            print '    mean percent improvement: %f' % oob_means
            print '    variables with >50 percent improvment: %f' % percent_majority
            print '    difference from previous imputation: %f' % conv_new

    drop_cols = [j for i in dummies.values() for j in i]
    ximp = ximp.drop(drop_cols, axis=1)
    
    if error_threshold is not None:
        cols_original = ximp.columns
        cols_percents = oob_error.iloc[:,-1]
        keep_imps = cols_percents>error_threshold
        if np.mean(keep_imps)<1:
            cols_imp = cols_percents.index[keep_imps]
            cols_mean = cols_percents.index[~keep_imps]
            cols_other = ximp.columns[~ximp.columns.isin(cols_percents.index)]
            x_imp = ximp[cols_imp].copy()
            x_mean = xmis[cols_mean].copy()
            x_other = ximp[cols_other].copy()

            if error_threshold > 0:
                if grouping_variables is not None:
                    x_mean = x_mean.groupby(grouping_index, sort=False)
                    x_mean = x_mean.fillna(x_mean.mean())
                else:
                    x_mean = x_mean.fillna(x_mean.mean()).copy()

            ximp = pd.concat([x_imp, x_mean, x_other], axis=1)
            ximp = ximp[cols_original]
            
            print 'mean imputation used for the following variables: %s' % (
                ', '.join([x for x in cols_mean]))
        else:
            print 'imputed values better than mean for all columns'
    
    df_list = [cols_from_dummies, ximp]

    if ignore_cols is not None:
        df_list.append(ignored_df)

    if (percent_missing==1.0).sum() > 0:
        df_list.append(xmis[cols_removed])

    ximp = pd.concat(df_list, axis=1)

    class missForest(object):
        random_forest = rfr
        oob_error_values = oob_error
        imputation_difference = conv_new
        importance_values = importance_scores
        kept_imputations = x_imputations
        output_data = ximp

    return missForest

def miss_dummies(xmis, verbose=False, cols_to_dummies=None, replace_with='mean',
    leave_cols=None, ignore_cols=None, grouping_variables=None, 
    summary_col=None, summary_covariates=None, summary_groups=None):
    '''

    Parameters
    ----------
    xmis : data matrix with missing values
    verbose : (boolean) if TRUE then missForest returns error estimates,
        runtime and if available true error during iterations
    cols_to_dummies : columns of xmis to turn into dummy variables before
        imputation
    ignore_cols : columns for which missing values should not be imputed; these
        will still have missing values filled in with column means and be
        included in the imputation of missing values for other columns; columns
        with no missing values do not need to be included in this list
    grouping_variables : columns names of variables to use as groups when
        calcluating averages and mean absolute scaled error

    '''

    xmis = xmis.replace([-np.inf, np.inf], np.nan)

    if grouping_variables is not None:
        grouping_index = xmis.set_index(grouping_variables).index

    if summary_groups is not None:
        summary_index = xmis.set_index(summary_groups).index

    if cols_to_dummies is not None:
        if verbose > 0:
            print 'creating dummy variables'
        dummies = []
        cols_from_dummies = xmis[cols_to_dummies].copy().astype('object')
        for col in cols_to_dummies:
            use_dummy_na = xmis[col].isnull().sum() > 0
            dummies.append(pd.get_dummies(xmis[col], prefix=col,
                prefix_sep='___', dummy_na=use_dummy_na))
        xmis = pd.concat([xmis] + dummies, axis=1)
        xmis = xmis.drop(cols_to_dummies, axis=1)
        dummies = dict(zip(cols_to_dummies,
            [list(x.columns.values) for x in dummies]))

    if ignore_cols is not None:
        ignored_df = xmis[ignore_cols]
        xmis = xmis.drop(ignore_cols, axis=1)

    ## extract missingness pattern
    if verbose > 0:
        print 'identifying missingness patterns'
    na_locations = xmis.isnull()
    percent_missing = na_locations.mean()
    percent_missing.sort()
    xmis = xmis[percent_missing.index].copy()
    cols_to_impute = (percent_missing > 0) & (percent_missing < 1.0)
    cols_to_impute = percent_missing[cols_to_impute].index.values
    cols_to_impute = pd.Series(cols_to_impute)

    if leave_cols is not None:
        cols_to_impute = cols_to_impute[~cols_to_impute.isin(leave_cols)]
    
    ## remove completely missing variables
    if (percent_missing==1.0).sum() > 0:
        cols_removed = percent_missing[percent_missing == 1.0].index.values
        print 'removed variable(s) %s due to the missingness of all entries' % (
              ', '.join(cols_removed))
    if verbose > 0:
        print 'the following columns will be imputed: %s' % (
            ', '.join(cols_to_impute))
 
    ## perform initial S.W.A.G. on xmis (mean imputation)
    if verbose > 0:
        print 'filling in missing values with %s' % replace_with
        
    keep_cols = percent_missing.index.values[(percent_missing != 1.0).values]
    ximp = xmis[keep_cols].copy()

    if grouping_variables is not None:
        ximp = ximp.groupby(grouping_index, sort=False)
        if replace_with=='mean':
            replacer = ximp.mean()
        elif replace_with=='zero':
            replacer = 0
    else:
        if replace_with=='mean':
            replacer = ximp.mean()
        elif replace_with=='zero':
            replacer = 0
    ximp = ximp.fillna(replacer).copy()

    missingness_dummies = na_locations[cols_to_impute].astype('float64')
    missingness_dummies.columns = [x+'___missing' for x in 
            missingness_dummies.columns]

    if summary_covariates is not None:
        if verbose > 0:
            print 'calculating summary covariates'
        index_names = summary_index.names
        add_index = pd.DataFrame(summary_index.tolist(), 
            columns=index_names)
        ximp = pd.concat([ximp, add_index], axis=1)
        ximp = wrappers.convenience.summary_covariates(ximp, 
            cols=[summary_col], group_cols=summary_groups, sep_string='___', 
            funcs=summary_covariates)
        ximp = ximp.drop(index_names, axis=1)

    df_list = [cols_from_dummies, ximp, missingness_dummies]

    if ignore_cols is not None:
        df_list.append(ignored_df)

    if (percent_missing==1.0).sum() > 0:
        df_list.append(xmis[cols_removed])

    ximp = pd.concat(df_list, axis=1)

    return ximp
