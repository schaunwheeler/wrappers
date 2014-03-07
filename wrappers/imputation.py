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
    keep_imputations=False, ignore_cols=None, grouping_variables=None):
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
        grouping_index = xmis[grouping_variables]

        if grouping_index.__class__ is pd.DataFrame:
           grouping_index = grouping_index.applymap(str).apply(sum, axis=1)

    if cols_to_dummies is not None:
        dummies = []
        cols_from_dummies = xmis[cols_to_dummies].copy()
        for col in cols_to_dummies:
            dummies.append(pd.get_dummies(xmis[col], prefix=col,
                prefix_sep='___', dummy_na=True))
        xmis = pd.concat([xmis] + dummies, axis=1)
        xmis = xmis.drop(cols_to_dummies, axis=1)
        dummies = dict(zip(cols_to_dummies,
            [list(x.columns.values) for x in dummies]))

    if(cores == "auto"):
        if(multiprocessing.cpu_count() > 1):
            cores = multiprocessing.cpu_count() - 1
        else:
            cores = 1

    ## extract missingness pattern
    na_locations = xmis.apply(lambda x: x.isnull())
    percent_missing = na_locations.mean()
    percent_missing.sort()
    xmis = xmis[percent_missing.index].copy()
    cols_to_impute = (percent_missing > 0) & (percent_missing < 1.0)
    cols_to_impute = percent_missing[cols_to_impute].index.values
    cols_to_impute = pd.Series(cols_to_impute)

    if ignore_cols is not None:
        cols_to_impute = cols_to_impute[~cols_to_impute.isin(ignore_cols)]

    ## remove completely missing variables
    if (percent_missing==1.0).sum() > 0:
        cols_removed = percent_missing[percent_missing == 1.0].index.values
        print 'removed variable(s) %s due to the missingness of all entries' % (
              ', '.join(cols_removed))

    ## perform initial S.W.A.G. on xmis (mean imputation)
    keep_cols = percent_missing.index.values[(percent_missing != 1.0).values]
    ximp = xmis[keep_cols].copy()

    if grouping_variables is not None:
        ximp = ximp.groupby(grouping_index, sort=False)
        ximp = ximp.fillna(ximp.mean())
    else:
        ximp = ximp.fillna(ximp.mean()).copy()

    ## initialize parameters of interest
    iteration = 1
    conv_new = 0
    conv_old = np.inf
    oob_error = pd.DataFrame(index=cols_to_impute)

    ## function to yield the stopping criterion in the following 'while' loop
    def stop_criterion(conv_new, conv_old, iteration, maxiter):
      output = (conv_new < conv_old) & (iteration <= maxiter)
      return output

    # function to yield mean absolute scaled error
    def mase(x, y, grouping=None):
        absolute_error = np.abs(x - y)
        if grouping is not None:
            naive_mae = pd.Series(x).groupby(grouping).transform(
                lambda x: np.mean(np.abs(x - np.mean(x)))).values
        else:
            naive_mae = np.mean(np.abs(x - np.mean(x)))
        ase = absolute_error / naive_mae
        ase[ase==np.inf] = np.nan
        ase[ase==-np.inf] = np.nan
        ase = ase[~np.isnan(ase)]
        output = np.mean(ase)
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
                grp_index = grouping_index.ix[observed]
                oob_error_val = wrappers.convenience.mase(obs_y.values,rfr.oob_prediction_,
                    grouping=grp_index.values)
                oob_error.loc[oob_error.index==col, iteration_name] = oob_error_val
            else:
                oob_error_val = wrappers.convenience.mase(obs_y.values,rfr.oob_prediction_)
                oob_error.ix[oob_error.index==col, iteration_name] = oob_error_val

            # predict missing parts of Y
            mis_y = rfr.predict(mis_x)
            ximp.ix[~observed, col] = mis_y

        if keep_imputations:
            x_imputations.append(ximp)

        ## check the difference between iteration steps
        conv_new = ((ximp[cols_to_impute] - ximp_old[cols_to_impute])**2 /
            (ximp[cols_to_impute]**2).sum()).sum().sum()

        ## return status output, if desired
        if verbose > 0:
            oob_error_mean = oob_error[iteration_name].mean()
            oob_error_min = oob_error[iteration_name].min()
            oob_error_max = oob_error[iteration_name].max()
            print ''
            print '    mean estimated error: %f' % oob_error_mean
            print '    minimum estimated error: %f' % oob_error_min
            print '    maximum estimated error: %f' % oob_error_max
            print '    difference from previous imputation: %f' % conv_new

    drop_cols = [j for i in dummies.values() for j in i]
    ximp = ximp.drop(drop_cols, axis=1)
    df_list = [cols_from_dummies, ximp]

    if (percent_missing==1.0).sum() > 0:
        df_list += [xmis[cols_removed]]

    ximp = pd.concat(df_list, axis=1)

    class missForest(object):
        random_forest = rfr
        oob_error_values = oob_error
        imputation_difference = conv_new
        kept_imputations = x_imputations
        output_data = ximp

    return missForest
