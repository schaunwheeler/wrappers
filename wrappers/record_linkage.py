#!/usr/bin/python
# -*- coding: utf-8 -*-

import pandas as pd
import numpy as np
import Levenshtein
import fuzzy
import wrappers.convenience
import statsmodels.api
#import dedupe

pd.options.mode.chained_assignment = None 

def identical_sim(field_1, field_2) :
    if field_1 and field_2 :
        if field_1 == field_2 :
            return 1
        else:
            return 0
    else :
        return np.nan

def absolute_sim(field_1, field_2):
    similarity = np.abs(field_1 - field_2)
    return similarity

def log1p_sim(field_1, field_2):
    similarity = np.log1p(abs(field_1 - field_2))
    return similarity

def percent_sim(field_1, field_2):
    similarity = abs(field_1 - field_2) / field_1
    return similarity

def inverse_absolute_sim(field_1, field_2):
    similarity = 1 / (abs(field_1 - field_2) + 1)
    return similarity

def inverse_log1p_sim(field_1, field_2):
    similarity = 1 / (np.log1p(abs(field_1 - field_2)) + 1)
    return similarity

def jarowinkler_sim(field_1, field_2):
    similarity = Levenshtein.jaro_winkler(field_1, field_2)
    return similarity

def scaledlevenshtein_sim(field_1, field_2):
    similarity = Levenshtein.ratio(field_1, field_2)
    return similarity

def metaphone_sim(field_1, field_2):
    f_1 = [fuzzy.DMetaphone()(x) for x in field_1.split(' ')]
    f_2 = [fuzzy.DMetaphone()(x) for x in field_2.split(' ')]    

    length = len(f_1) - len(f_2)
    if length < 0:    
        f_1 += ([['___1', '___2']] * np.abs(length))
    elif length > 0:    
        f_2 += ([['___1', '___2']] * np.abs(length))        
    
    f_zipped = zip(f_1, f_2)   
    f_sets = [x+y for (x,y) in f_zipped]

    for i in range(len(f_sets)):
        replacement = [x for x in f_sets[i] if x is not None]
        if replacement == []:
            replacement = ['None1', 'None2', 'None1', 'None2']
        f_sets[i] = replacement
    
    similarities = [(len(x) - len(set(x))) / len(set(x)) for x in f_sets]
    similarity = np.mean(similarities)
    return similarity

def jaccard_sim(list_1, list_2, fillna=1):
    
    if (list_1 == []) & (list_2 == []):
        similarity = fillna
    else:
        set_1 = set(list_1)
        set_2 = set(list_2)
        n = len(set_1.intersection(set_2))
        similarity = n / float(len(set_1) + len(set_2) - n)
    return similarity

def return_same(value):
    return value

def return_metaphone(value):
     value = [fuzzy.DMetaphone()(x) for x in value.split(' ')]
     value = [x for y in value for x in y if x is not None]
     value = ''.join(value)
     return value

identical_simv = np.vectorize(lambda x: identical_sim(*x))
absolute_simv = np.vectorize(lambda x: absolute_sim(*x))
log1p_simv = np.vectorize(lambda x: log1p_sim(*x))
percent_simv = np.vectorize(lambda x: percent_sim(*x))
inverse_absolute_simv = np.vectorize(lambda x: inverse_absolute_sim(*x))
inverse_log1p_simv = np.vectorize(lambda x: inverse_log1p_sim(*x))
jarowinkler_simv = np.vectorize(lambda x: jarowinkler_sim(*x))
scaledlevenshtein_simv = np.vectorize(lambda x: scaledlevenshtein_sim(*x))
metaphone_simv = np.vectorize(lambda x: metaphone_sim(*x))
jaccard_simv = np.vectorize(lambda x: jaccard_sim(*x))

identical_simv.__name__ = 'identical_simv'
absolute_simv.__name__ = 'absolute_simv'
log1p_simv.__name__ = 'log1p_simv'
percent_simv.__name__ = 'percent_simv'
inverse_absolute_simv.__name__ = 'inverse_absolute_simv'
inverse_log1p_simv.__name__ = 'inverse_log1p_simv'
jarowinkler_simv.__name__ = 'jarowinkler_simv'
scaledlevenshtein_simv.__name__ = 'scaledlevenshtein_simv'
metaphone_simv.__name__ = 'metaphone_simv'
jaccard_simv.__name__ = 'jaccard_simv'

def dataframe_linkage2(records, settings, blocking, id_col=None, weights=None,
    reset_index=True, threshold=None, keep_best=True, verbose=True):
    '''
    records : data frame or list of data frames to convert to dictionaries;
        columns to be used in records linkage must be consistently named across
        data frames; if a list or dictionary or two dataframes is passed to
        records, the two dataframes will be linked. Otherwise, the function
        assumes a single dataframe and will label duplicates
    settings : file or dictionary containing settings to pass to dedupe module
    blocking : column or list of columns on which to block records
    id_col: name of column containing unique identifiers; id_col, 
        like all columns, must be consistent across data frames
    reset_index : reset index of data frames as a precaution to make sure each
        row has a unique identifier
    threshold : float between 0.0 and 1.0 indicating cutoff point below which
        words will not be considered matches
    keep_best : integer, when multiple potential matches are found for a word, 
        the top keep_best will be kept
    verbose : boolean, indicating whether to output informative messages during
        the deduping process

    Process
    _______
    
    1. run with threshold==0.0 and keep_best==2
    2. Get pairs
    3. Randomly select pairs and evaluate them
        a. Use input plus exact matches to determine threshold
        b. Perform logistic regression on labelled data to weight dict keys
    4. re-run with threshold=thredhold, keep_best=True



    '''
    
    settings = dict(settings)    
    
    check_dicts = np.mean([type(v)==dict for (k,v) in settings.items()])

    if check_dicts==0:
        if weights is not None:
            for (k,v) in settings.items():
                settings[k] = {'function':v, 'weight': weights[k][v.__name__]}
        else:
            settings = {k:{'function':v, 'weight':1} for (k,v) in settings.items()}
    elif check_dicts==1:
        weights = True
    else:
        raise Exception('inconsistent value types in settings dictionary')
        
    if type(blocking) is not dict:
        blocking = {blocking:return_same}
        
    record_type = type(records)
    
    if verbose:
        print 'preparing records for linkage...'
    
    if record_type == dict:
        input_ids = ['_'+(x) for x in records.keys()]
        records = [x for x in records.values()]
    elif record_type == list:
        input_ids = ['_first', '_second']
    elif record_type == pd.DataFrame:
        input_ids = [None]
        records = [records]

    if reset_index:
        records = [x.reset_index(drop=True) for x in records]
    
        if id_col is None:
            id_col = 'id'
            for i in range(len(records)):
                records[i]['id'] = [str(x) for x in records[i].index]
            
    for i in range(len(records)):
        blocker = records[i][blocking.keys()]
        records[i].columns = records[i].columns.values + input_ids[i]

        for key in blocking.keys():
            blocker[key] = blocker[key].apply(blocking[key]).copy()
        
        if blocker.shape[1] > 1:
            records[i]['blocker'] = blocker.fillna('').applymap(str).apply(
                lambda x: ''.join(x), axis=1)
        else:
            records[i]['blocker'] = blocker.squeeze()
    
    blocker = set.intersection(*[set(x['blocker'].unique()) for x in records])
    blocker = list(blocker)
    
    matched_df = pd.DataFrame()
    
    if verbose:
        print 'matching in progress...'
    
    target_columns = settings.keys()
    
    for block in blocker:
        if verbose:
            print 'processing block %d of %d - columns (out of %d) completed:' % (
                blocker.index(block) + 1, len(blocker), len(target_columns)) 
        temp_records = [x[x['blocker']==block].copy() for x in records]
        temp_records = pd.merge(temp_records[0], temp_records[1], 
            how='outer', on ='blocker', suffixes = input_ids) 
        temp_cols = temp_records.columns
        id_columns = [x for x in temp_cols if x.startswith(id_col)]
        results = temp_records[id_columns]

        for col in target_columns:
            wanted_cols = [col+x for x in input_ids]
            pairs = temp_records[wanted_cols].to_records(index=False)
            colname = col + '__' + settings[col]['function'].__name__
            results[colname] = settings[col]['function'](pairs)
            if verbose:
                print '%d ' % (target_columns.index(col) + 1),
        
        weights_list = [x['weight'] for x in settings.values()]
        column_order = [k + '__' + v['function'].__name__ for (k,v) in settings.items()]
        means = results.set_index(id_columns)[column_order]
        means = means * weights_list
        means = means.sum(axis=1) / sum(weights_list)
        means = means.reset_index(drop=True)
        
        results['mean_similarity'] = means

        if threshold is not None:        
            
            keeps = (means > threshold).values
            results = results[keeps]
        
        if keep_best:
            
            results = results.sort(columns='mean_similarity', ascending=False)
            results = results.groupby(id_columns[0], as_index=False).head(keep_best)

            results = results.sort(columns='mean_similarity', ascending=False)
            results = results.groupby(id_columns[1], as_index=False).head(keep_best)

        matched_df = matched_df.append(results, ignore_index=True)
        print ''

    matched_df['cluster_id'] = range(matched_df.shape[0])

    records[0] = pd.merge(records[0], matched_df, how='left', on=id_columns[0])
    records[1] = pd.merge(records[1], matched_df, how='left', on=id_columns[1])
        
    records[0] = records[0].rename(columns={id_columns[0]: id_col, 
        id_columns[1]: id_col+'_match'})
    records[1] = records[1].rename(columns={id_columns[1]: id_col, 
        id_columns[0]: id_col+'_match'})

    records[0].columns = [x.replace(input_ids[0], '') for x in 
        records[0].columns]
    records[1].columns = [x.replace(input_ids[1], '') for x in 
        records[1].columns]
    
    start_cols = [id_col, id_col+'_match', 'cluster_id', 'blocker', 'mean_similarity']
    other_cols = [x for x in records[0].columns if x not in start_cols]    
    column_order = start_cols + other_cols    
    records = [x[column_order] for x in records]
    
    if record_type == dict:
        records = {input_ids[i].replace('_', ''):records[i] for i in 
            range(len(input_ids))}
    
    return records


def dataframe_linkage(records, settings, blocking, id_col=None, weights=None,
    reset_index=True, threshold=None, keep_best=True, verbose=True):
    '''
    records : data frame or list of data frames to convert to dictionaries;
        columns to be used in records linkage must be consistently named across
        data frames; if a list or dictionary or two dataframes is passed to
        records, the two dataframes will be linked. Otherwise, the function
        assumes a single dataframe and will label duplicates
    settings : file or dictionary containing settings to pass to dedupe module
    blocking : column or list of columns on which to block records
    id_col: name of column containing unique identifiers; id_col, 
        like all columns, must be consistent across data frames
    reset_index : reset index of data frames as a precaution to make sure each
        row has a unique identifier
    threshold : float between 0.0 and 1.0 indicating cutoff point below which
        words will not be considered matches
    keep_best : integer, when multiple potential matches are found for a word, 
        the top keep_best will be kept
    verbose : boolean, indicating whether to output informative messages during
        the deduping process

    Process
    _______
    
    1. run with threshold==0.0 and keep_best==2
    2. Get pairs
    3. Randomly select pairs and evaluate them
        a. Use input plus exact matches to determine threshold
        b. Perform logistic regression on labelled data to weight dict keys
    4. re-run with threshold=thredhold, keep_best=True



    '''
    
    settings = dict(settings)    
    
    check_dicts = np.mean([type(v)==dict for (k,v) in settings.items()])

    if check_dicts==0:
        if weights is not None:
            for (k,v) in settings.items():
                settings[k] = {'function':v, 'weight': weights[k][v.__name__]}
        else:
            settings = {k:{'function':v, 'weight':1} for (k,v) in settings.items()}
    elif check_dicts==1:
        weights = True
    else:
        raise Exception('inconsistent value types in settings dictionary')
        
    if type(blocking) is not dict:
        blocking = {blocking:return_same}
        
    record_type = type(records)
    
    if verbose:
        print 'preparing records for linkage...'
    
    if record_type == dict:
        input_ids = ['_'+(x) for x in records.keys()]
        records = [x for x in records.values()]
    elif record_type == list:
        input_ids = ['_first', '_second']
    elif record_type == pd.DataFrame:
        input_ids = [None]
        records = [records]

    if reset_index:
        records = [x.reset_index(drop=True) for x in records]
    
        if id_col is None:
            id_col = 'id'
            for i in range(len(records)):
                records[i]['id'] = [str(x) for x in records[i].index]
            
    for i in range(len(records)):
        blocker = records[i][blocking.keys()]
        records[i].columns = records[i].columns.values + input_ids[i]

        for key in blocking.keys():
            blocker[key] = blocker[key].apply(blocking[key]).copy()
        
        if blocker.shape[1] > 1:
            records[i]['blocker'] = blocker.fillna('').applymap(str).apply(
                lambda x: ''.join(x), axis=1)
        else:
            records[i]['blocker'] = blocker.squeeze()
    
    blocker = set.intersection(*[set(x['blocker'].unique()) for x in records])
    blocker = list(blocker)
    
    matched_df = pd.DataFrame()
    
    if verbose:
        print 'matching in progress...'
    
    target_columns = settings.keys()
    
    for block in blocker:
        if verbose:
            print 'processing block %d of %d - columns (out of %d) completed:' % (
                blocker.index(block) + 1, len(blocker), len(target_columns)) 
        temp_records = [x[x['blocker']==block].copy() for x in records]
        temp_records = pd.merge(temp_records[0], temp_records[1], 
            how='outer', on ='blocker', suffixes = input_ids) 
        temp_cols = temp_records.columns
        id_columns = [x for x in temp_cols if x.startswith(id_col)]
        results = temp_records[id_columns]

        for col in target_columns:
            wanted_cols = [col+x for x in input_ids]
            pairs = temp_records[wanted_cols].to_records(index=False)
            colname = col + '__' + settings[col]['function'].__name__
            results[colname] = settings[col]['function'](pairs)
            if verbose:
                print '%d ' % (target_columns.index(col) + 1),
        
        weights_list = [x['weight'] for x in settings.values()]
        column_order = [k + '__' + v['function'].__name__ for (k,v) in settings.items()]
        means = results.set_index(id_columns)[column_order]
        means = means * weights_list
        means = means.sum(axis=1) / sum(weights_list)
        means = means.reset_index(drop=True)
        
        results['mean_similarity'] = means

        if threshold is not None:        
            
            keeps = (means > threshold).values
            results = results[keeps]
        
        if keep_best:
            
            results = results.sort(columns='mean_similarity', ascending=False)
            results = results.groupby(id_columns[0], as_index=False).head(keep_best)

            results = results.sort(columns='mean_similarity', ascending=False)
            results = results.groupby(id_columns[1], as_index=False).head(keep_best)

        matched_df = matched_df.append(results, ignore_index=True)
        print ''

    matched_df['cluster_id'] = range(matched_df.shape[0])

    records[0] = pd.merge(records[0], matched_df, how='left', on=id_columns[0])
    records[1] = pd.merge(records[1], matched_df, how='left', on=id_columns[1])
        
    records[0] = records[0].rename(columns={id_columns[0]: id_col, 
        id_columns[1]: id_col+'_match'})
    records[1] = records[1].rename(columns={id_columns[1]: id_col, 
        id_columns[0]: id_col+'_match'})

    records[0].columns = [x.replace(input_ids[0], '') for x in 
        records[0].columns]
    records[1].columns = [x.replace(input_ids[1], '') for x in 
        records[1].columns]
    
    start_cols = [id_col, id_col+'_match', 'cluster_id', 'blocker', 'mean_similarity']
    other_cols = [x for x in records[0].columns if x not in start_cols]    
    column_order = start_cols + other_cols    
    records = [x[column_order] for x in records]
    
    if record_type == dict:
        records = {input_ids[i].replace('_', ''):records[i] for i in 
            range(len(input_ids))}
    
    return records

def get_pairs(df1, df2, id_col, match_col, comparison_cols, metric_cols=None,
              append_nomatches=True, flag_matches=None):

    if metric_cols is None:
        metric_cols = []
    elif type(metric_cols) is str:
        metric_cols = [metric_cols]

    df1_comparison = df1[[id_col] + comparison_cols]
    df2_comparison = df2[[id_col] + comparison_cols]

    if append_nomatches:    
        df1_nomatch = df1[df1[match_col].isnull()]
        df2_nomatch = df2[df2[match_col].isnull()]
        
    df = df2[df2[match_col].notnull()]
    df = df.sort(columns=id_col)
    df = df[[id_col, match_col] + metric_cols]

    df = pd.merge(df, df2_comparison, how='left', left_on=id_col, 
        right_on=id_col)
    df = df.rename(columns={id_col:id_col+'_original'})
    df = pd.merge(df, df1_comparison, how='left', left_on=match_col, 
        right_on=id_col, suffixes=('', '_match'))
    df = df.drop([id_col], axis=1)
    df = df.rename(columns={id_col+'_original':id_col})
    df = df.drop_duplicates(cols=[id_col, match_col])
    
    if append_nomatches:
        df1_nomatch = df1_nomatch[[id_col, match_col]]
        df2_nomatch = df2_nomatch[[id_col, match_col]]
        df2_nomatch = df2_nomatch.rename(columns={id_col:match_col, match_col:id_col})
    
        df = pd.concat([df, df1_nomatch, df2_nomatch], ignore_index=True)
    
    col_order = [id_col, match_col] + metric_cols + comparison_cols + \
        [x + '_match' for x in comparison_cols]
        
    df = df[col_order]

    if flag_matches is not None:
        df['true_match'] = np.nan
        df['true_match'][df[flag_matches]==1] = 1.0
        true_match_ids = df['id'][df[flag_matches]==1]
        false_matches = (df[flag_matches]!=1) & df['id'].isin(true_match_ids)
        df['true_match'][false_matches] = 0.0

    return df

def tune_linkage(data, output_col, train_cols, k, similarity_metric, 
    naive_prediction=None, recall_cutoff=.95, boot=250):

    folds = np.random.choice(k,data.shape[0])
    errors = []
    weights = pd.DataFrame()
    predictions = pd.DataFrame()
    
    for fold in sorted(np.unique(folds)):
        y_train = data[output_col][folds!=fold]
        x_train = data[train_cols][folds!=fold]
        x_test = data[train_cols][folds==fold]
        y_test = data[output_col][folds==fold]
        
        value_counts = y_train.value_counts()
        if len(value_counts)>2:
            print 'fold does not contain both true and false matches'
    
        logit = statsmodels.api.Logit(y_train, x_train)
        result = logit.fit()
        weights = pd.concat([weights, result.params.sort_index().to_frame().T], 
            ignore_index=True)
    
        if naive_prediction is None:
            naive_prediction = value_counts.max() / value_counts.sum()
    
        model_prediction = result.predict(x_test)
        
        predicts = pd.DataFrame({'original_data':y_test.values, 
            'model_predictions':model_prediction,
            'similarity':data[similarity_metric][folds==fold]})
        predicts['naive_predicton'] = naive_prediction
        predicts['fold'] = fold
        
        predictions = pd.concat([predictions, predicts], ignore_index=True)
        
        model_error = wrappers.convenience.mase(model_prediction, y_test, 
             func=lambda z: np.mean(np.abs(z - naive_prediction)))
        errors.append(model_error)
    
    mean_weights = weights[train_cols].apply(
        lambda x: np.abs(np.average(x, weights=errors)), axis=0)
    mean_weights = mean_weights.to_dict()
    mean_weights = {k.split('__')[0]:{k.split('__')[1]:v} for (k,v) 
        in mean_weights.items()}
        
    fps = predictions[predictions['similarity']!=1.0]['similarity']
    fps_shape = fps.shape[0]
    boot_samples = []
    for i in range(boot):
        sampled = fps.iloc[np.random.choice(fps_shape, fps_shape)]
        sampled = sampled.quantile(recall_cutoff)
        boot_samples.append(sampled)
    
    threshold = np.mean(boot_samples)

    class results():
        def __init__(self, weights, mean_weights, errors, predictions, threshold):
            self.weights=weights
            self.mean_weights=mean_weights
            self.errors=errors
            self.predictions=predictions
            self.threshold=threshold
    
    output = results(weights, mean_weights, errors, predictions, threshold)

    return output


def matched_preprocessing(df1, df2, group_cols, index_cols, threshold=None,
    k=1, combine=False, verbose=False):

    data1 = df1.copy()
    data2 = df2.copy()
    
    other_cols = [x for x in data1.columns if x not in (group_cols+index_cols)]
    data1 = data1.set_index(index_cols+group_cols)
    data2 = data2.set_index(index_cols+group_cols)
    
    data1 = data1.unstack(group_cols)
    data2 = data2.unstack(group_cols)
    
    final = pd.DataFrame(data=0.0, columns=data1.columns.droplevel(0).unique(),
        index=data2.columns.droplevel(0).unique())
    
    if verbose:
        print 'calculating correlations'    
    
    for x in other_cols:
        if verbose:
            print x
        mat1 = data1.ix[:,data1.columns.get_level_values(0)==x]
        mat2 = data2.ix[:,data2.columns.get_level_values(0)==x]
        output = pd.DataFrame(data=0.0, columns=mat1.columns, index=mat2.columns)
        weights = pd.Series(data=0.0, index=mat2.columns)
        for lev in mat1.columns:
            if verbose:
                print len(mat1.columns.values) - list(mat1.columns.values).index(lev),
            weights += (mat1[lev].notnull().mean() * mat2.notnull().mean())
            corrs = (mat2.corrwith(mat1[lev]).abs() * weights)
            corrs = corrs.fillna(0)
            output.loc[:,lev] = corrs
        output = output.div(weights, axis='index')
        output.columns = output.columns.droplevel(0)
        output.index = output.index.droplevel(0)
        final += output
        if verbose:
            print ' '
    
    final = (final / len(other_cols))
    final.index = pd.MultiIndex.from_tuples(final.index, names=group_cols)
    final.columns = pd.MultiIndex.from_tuples(final.columns, names=group_cols)
    
    final = final.stack(group_cols)
    final.name = 'value'
    match_cols = [x+'_match' for x in group_cols]
    final.index.names = match_cols + group_cols
    final = final.reset_index()
    
    final = final.sort(columns=['value'], ascending=False).reset_index(drop=True)
    
    if threshold is not None:
        final = final[final['value']>threshold]
    
    if k is not None:
        if verbose:
            print 'identifying nearest neighbors'
        hash_table = final[group_cols].drop_duplicates()
        
        for it in range(k):
            if verbose:
                print 'iteration: %d' % (it+1)
            for i in range(hash_table.shape[0]):
                if verbose:
                    print hash_table.shape[0] - range(hash_table.shape[0]).index(i),  
                flag = (final[group_cols] == hash_table.iloc[i,:])
                flag = flag.sum(axis=1) == len(group_cols)
                match_flag = pd.Series([(x,y) for x,y in final[match_cols].values])
                match_flag = match_flag.isin(
                    [(x,y) for x,y in final.ix[flag, match_cols].head(it+1).values])
                flag_first = (flag.index[flag])[:(it+1)]
                flag_first = ~flag.index.isin(flag_first)
                final = final[~(match_flag.values & flag_first)]
                if verbose:
                    print ''
        
        final = final.groupby(group_cols).head(k)
        final = final.reset_index(drop=True)
    
        if combine:
            if verbose:
                print 'combining data sets'
            data1 = data1.stack(group_cols).reset_index()
            data2 = data2.stack(group_cols).reset_index()
            data2_values = pd.Series([(x,y) for x,y in data2[group_cols].values])
            matched_values = [(x,y) for x,y in final[match_cols].values]
            data2 = data2[data2_values.isin(matched_values)]
            final = pd.concat([data1, data2])
    
    return final

# Function to convert data frame to hashable dictonary
#def dataframe_to_recorddict(df, df_tag=None, id_col=None, to_string=False):
#    '''
#    df : data frame to convert to a frozen dictionary
#    df_tag : an identifier for the data frame itself, to append to id values
#    id_col : column of unique identifiers; if None (default), the data frame
#        index will be used
#    to_string : boolean, indicating whether to convert all values to type str
#
#    '''
#
#    if id_col is not None:
#        id_values = df[id_col].values
#    else:
#        id_values = df.reset_index().index.values
#
#    if df_tag is not None:
#        id_values = [df_tag + str(x) for x in id_values]
#
#    colnames = df.columns.values
#
#    df = df.fillna('')
#
#    if to_string:
#        df = df.applymap(str)
#
#    df= df.groupby(id_values)
#    df = df.apply(lambda x: dict(zip(colnames, x.values[0])))
#    df.index = [x for x in df.index.values]
#
#    df_d = df.to_dict()
#
#    return df_d
#
#
#def group_by_exact_matches(dataframe, columns, group_label='group', tag=None):
#    '''
#    Group records that are exact duplicates on specified columns.
#
#    dataframe : the data frame to be split
#    columns : columns along which to look for duplicate values
#    group_label : the name to assign to the column containing duplicate values
#    tag : an optional tag to append to group assigments
#
#
#    '''
#
#    if type(columns) == str:
#        columns = [columns]
#
#    if type(dataframe) == list:
#        df = dataframe[0].copy()
#    else:
#        df = dataframe.copy()
#
#    topdown = df[columns].apply(
#        lambda x: x.duplicated(take_last=False))
#    bottomup = df[columns].apply(
#        lambda x: x.duplicated(take_last=True))
#    exact_matches = topdown | bottomup
#
#    duplicate_index = exact_matches.apply(np.sum, axis=1) > 0
#
#    if duplicate_index.sum() > 0:
#        duplicates = df[duplicate_index]
#        not_duplicates = df[~duplicate_index]
#
#        replacements = pd.Series(
#            duplicates.groupby(columns).grouper.group_info[0],
#            index=duplicates.index)
#
#        if tag is not None:
#            replacements = replacements.apply(str) + tag
#
#        duplicates[group_label] = replacements
#
#        df = pd.concat([duplicates, not_duplicates])
#    else:
#        df[group_label] = None
#
#    return df
#
#
# def dataframe_linkage(records, settings, id_col=None, to_string=False,
#     reset_index=True, records_sample=0.01, training_file=None, training='append',
#     settings_output=None, threshold=None, recall_weight=1.5,
#     split_output=False, override_columns=None, exclude_exact_matches=True,
#     ppc=1, uncovered_dupes=1, verbose=True):
#     '''
#     records : data frame or list of data frames to convert to dictionaries;
#         columns to be used in records linkage must be consistently named across
#         data frames; if a list or dictionary or two dataframes is passed to
#         records, the two dataframes will be linked. Otherwise, the function
#         assumes a single dataframe and will label duplicates
#     settings : file or dictionary containing settings to pass to dedupe module
#     id_col: name of column containing unique identifiers (passed to
#         `dataframe_to_frozendict`); id_col, like all columns, must be consistent
#         across data frames
#     to_string : boolean, indicating whether to convert all values to type str;
#         passed to `dataframe_to_frozendict`
#     reset_index : reset index of data frames as a precaution to make sure each
#         row has a unique identifier
#     records_sample : number of random pairs to pull from data for training; if less
#         than 1, sample size will equal sample size times the number of possible
#         combinations of all rows across data frames
#     training_file : if not None, the location of of a json file containing already-
#         categorized pairs
#     training : 'replace', 'append', or 'none' indicating whether to replace any
#         existing training files with new training data, to append new training
#         data, or to skip training altogether
#     block_sample : float betwen 0.0 and 1.0, indicating the percentage of blocked
#         data to sample in determining a good threshold for clustering
#     recall_weight : weight to determin threshold to maximize a weighted average of
#         precision and recall (recall_weight==2 means we care twice as much about
#         recall as we do precision)
#     threshold : float between 0.0 and 1.0 indicating cutoff point in logistic
#         regression predictons for determining matches. If None, threshold will
#         be determined algorithmically.
#     split_output : boolean, indicating whether to return a list of data frames
#         corresponding to the structure of the input list; if split_output==False,
#         a single data frame will be returned with a 'data_frame_id' column
#         indicating which records belonged to which item of the input list; if
#         only one data frame is input, this parameter will be ignored
#     override_columns : a columns of list of columns that automatically trigger
#         a match; if two records match on all override_columns, they are labelled
#         as a match and are not processed through the dedupe algorithms
#     exclude_exact_matches : boolean, indicating whether to leave exact matches
#         from override_columns out of the records for fuzzy matching
#     ppc : float between 0.0 and 1.0; the proportion of all possible pairs a
#         predicate is allowed to cover. If a predicate puts together a fraction
#         of possible pairs greater than ppc, it will be removed from consideration.
#         Passed to dedupes `blockingFuncton`
#     uncovered_dupes : integer, the number of true dupes pairs in the training
#         that can fail to be placed in a block. Passed to dedupes `blockingFunction`
#     verbose : boolean, indicating whether to output informative messages during
#         the deduping process


#     '''

#     if type(settings) is dict:
#         settings = dict(settings)

#     if training_file is not None:
#         if (not os.path.exists(training_file)) & (training == 'none'):
#             raise Exception('must supply training_file or do active training')
#         elif os.path.exists(training_file) & (training == 'replace'):
#             os.remove(training_file)

#     record_type = type(records)

#     # Convert all data frames to frozen dictionaries
#     if verbose:
#         print 'preparing records for linkage...'

#     if record_type == dict:
#         input_ids = [(x+'_') for x in records.keys()]
#         records = [x for x in records.values()]
#     elif record_type == list:
#         input_ids = ['first_', 'second_']
#     elif record_type == pd.DataFrame:
#         input_ids = [None]
#         records = [records]


#     if override_columns is not None:

#         records = pd.concat(records, keys=input_ids)

#         records = group_by_exact_matches(dataframe=records,
#             columns=override_columns, group_label='cluster_id', tag='_exact')

#         if exclude_exact_matches:
#             exact_dupes = records[records['cluster_id'].notnull()]
#             records = records[records['cluster_id'].isnull()]

#         if len(input_ids) == 2:
#             records = [records[records.index.get_level_values(0) == x].copy()
#                 for x in input_ids]
#             exact_dupes = [exact_dupes[exact_dupes.index.get_level_values(0) == x].copy()
#                 for x in input_ids]
#         else:
#             records = [records]
#             exact_dupes = [exact_dupes]

#     if id_col is not None:
#         records = [x.set_index(id_col, drop=False) for x in records]
#     elif reset_index:
#         records = [x.reset_index(drop=True) for x in records]

#     for i in range(len(records)):
#         records[i].index = [str(x) for x in records[i].index]

#     record_dicts = []
#     for i in range(len(records)):
#         record_d = dataframe_to_recorddict(records[i], id_col=id_col,
#             df_tag = input_ids[i], to_string=to_string)
#         record_dicts.append(record_d)

#     if records_sample < 1:
#         n_dicts = len(record_dicts)

#         if n_dicts == 1:
#             n_records = sum([len(x) for x in record_dicts])
#             n_combinations = n_records * (n_records - 1) / 2
#         else:
#             n_combinations = np.product([len(x) for x in record_dicts])

#         records_sample = int(np.ceil(n_combinations * records_sample))

#     # Instantiate appropriate Dedupe class
#     if len(records) == 2:
#         linker = dedupe.api.RecordLink(settings)
#     else:
#         linker = dedupe.Dedupe(settings)

#     if os.path.exists(training_file):
#         if verbose:
#             print 'reading in existing training file...'
#         linker.readTraining(training_file)

#     if type(settings) is dict:

#         if training != 'none':
#             # Sample data for active learning

#             if verbose:
#                 print 'randomly sampling records to perform active labelling...'

#             if len(records) == 2:
#                 linker.sample(record_dicts[0], record_dicts[1], records_sample)
#             else:
#                 linker.sample(record_dicts[0], records_sample)

#             if verbose:
#                 print 'initiating active labeling...'

#             dedupe.consoleLabel(linker)

#         if verbose:
#             print 'training model...'

#         linker.train(ppc=ppc, uncovered_dupes=uncovered_dupes)

#         if training_file is not None:
#             linker.writeTraining(training_file)
#         if settings_output is not None:
#             linker.writeSettings(settings_output)

#         if threshold == None:
#             if verbose:
#                 print 'blocking...'
#             if len(records) == 2:
#                 threshold = linker.threshold(record_dicts[0], record_dicts[1],
#                     recall_weight=recall_weight)
#             else:
#                 threshold = linker.threshold(record_dicts[0],
#                     recall_weight=recall_weight)

#     if verbose:
#         print 'clustering...'

#     if len(records) == 2:
#         clustered_dupes = linker.match(record_dicts[0], record_dicts[1],
#             threshold)
#     else:
#         clustered_dupes = linker.match(record_dicts[0], threshold)

#     if verbose:
#         print '# duplicate sets', len(clustered_dupes)

#     if len(records)==2:
#         clustered_dupes = enumerate(clustered_dupes)
#         clustered_dupes = [(y, z, x) for (x,(y,z))  in clustered_dupes]
#         record_0_dupes = pd.Series({ str(x).replace(input_ids[0], ''):z for
#             (x,y,z) in clustered_dupes })
#         record_1_dupes = pd.Series({ str(y).replace(input_ids[1], ''):z for
#             (x,y,z) in clustered_dupes })
#         record_0_dupes = record_0_dupes.reindex(index=records[0].index)
#         record_1_dupes = record_1_dupes.reindex(index=records[1].index)

#         records[0]['cluster_id'] = record_0_dupes
#         records[1]['cluster_id'] = record_1_dupes

#         if (override_columns is not None) & exclude_exact_matches:
#             records[0] = pd.concat([records[0], exact_dupes[0]], ignore_index=True)
#             records[1] = pd.concat([records[1], exact_dupes[1]], ignore_index=True)

#     else:
#         # create a 'cluster_id' column to show which records fall within the same cluster
#         clustered_dupes = enumerate(clustered_dupes)
#         clustered_dupes = [zip(y, [x]*len(y)) for (x, y) in clustered_dupes]
#         clustered_dupes = [x for y in clustered_dupes for x in y]
#         cluster_membership = pd.Series({ x:y for (x,y) in clustered_dupes })
#         cluster_membership.index = [str(x) for x in cluster_membership.index]
#         cluster_membership = cluster_membership.reindex(index=records[0].index)

#         records['cluster_id'] = cluster_membership

#         if (override_columns is not None) & exclude_exact_matches:
#             records[0] = pd.concat([records[0], exact_dupes[0]], ignore_index=True)

#     if record_type == dict:
#         records = {input_ids[i].replace('_', ''):records[i] for i in range(len(input_ids))}
#     elif record_type == pd.DataFrame:
#         records = records[0]

#     return records
