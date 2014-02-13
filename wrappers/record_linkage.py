#!/usr/bin/python
# -*- coding: utf-8 -*-

import os
import pandas as pd
import numpy as np
import Levenshtein
#import dedupe

def identical_diff(field_1, field_2) :
    if field_1 and field_2 :
        if field_1 == field_2 :
            return 1
        else:
            return 0
    else :
        return np.nan

def absolute_diff(field_1, field_2):
    distance = np.abs(field_1 - field_2)
    return distance

def log1p_diff(field_1, field_2):
    distance = np.log1p(abs(field_1 - field_2))
    return distance

def percent_diff(field_1, field_2):
    distance = abs(field_1 - field_2) / field_1
    return distance

def inverse_absolute_diff(field_1, field_2):
    distance = 1 / (abs(field_1 - field_2) + 1)
    return distance

def inverse_log1p_diff(field_1, field_2):
    distance = 1 / (np.log1p(abs(field_1 - field_2)) + 1)
    return distance

def jarowinkler_diff(field_1, field_2):
    distance = Levenshtein.jaro_winkler(field_1, field_2)
    return distance

def scaledlevenshtein_diff(field_1, field_2):
    distance = Levenshtein.ratio(field_1, field_2)
    return distance

def levenshtein_diff(field_1, field_2):
    distance = Levenshtein.distance(field_1, field_2)
    return distance

identical_diffv = np.vectorize(lambda x: identical_diff(*x))
absolute_diffv = np.vectorize(lambda x: absolute_diff(*x))
log1p_diffv = np.vectorize(lambda x: log1p_diff(*x))
percent_diffv = np.vectorize(lambda x: percent_diff(*x))
inverse_absolute_diffv = np.vectorize(lambda x: inverse_absolute_diff(*x))
inverse_log1p_diffv = np.vectorize(lambda x: inverse_log1p_diff(*x))
jarowinkler_diffv = np.vectorize(lambda x: jarowinkler_diff(*x))
scaledlevenshtein_diffv = np.vectorize(lambda x: scaledlevenshtein_diff(*x))
levenshtein_diffv = np.vectorize(lambda x: levenshtein_diff(*x))

identical_diffv.__name__ = 'identical_diffv'
absolute_diffv.__name__ = 'absolute_diffv'
log1p_diffv.__name__ = 'log1p_diffv'
percent_diffv.__name__ = 'percent_diffv'
inverse_absolute_diffv.__name__ = 'inverse_absolute_diffv'
inverse_log1p_diffv.__name__ = 'inverse_log1p_diffv'
jarowinkler_diffv.__name__ = 'jarowinkler_diffv'
scaledlevenshtein_diffv.__name__ = 'scaledlevenshtein_diffv'
levenshtein_diffv.__name__ = 'levenshtein_diffv'

# Function to convert data frame to hashable dictonary
def dataframe_to_recorddict(df, df_tag=None, id_col=None, to_string=False):
    '''
    df : data frame to convert to a frozen dictionary
    df_tag : an identifier for the data frame itself, to append to id values
    id_col : column of unique identifiers; if None (default), the data frame
        index will be used
    to_string : boolean, indicating whether to convert all values to type str

    '''

    if id_col is not None:
        id_values = df[id_col].values
    else:
        id_values = df.reset_index().index.values

    if df_tag is not None:
        id_values = [df_tag + str(x) for x in id_values]

    colnames = df.columns.values

    df = df.fillna('')

    if to_string:
        df = df.applymap(str)

    df= df.groupby(id_values)
    df = df.apply(lambda x: dict(zip(colnames, x.values[0])))
    df.index = [x for x in df.index.values]

    df_d = df.to_dict()

    return df_d


def group_by_exact_matches(dataframe, columns, group_label='group', tag=None):
    '''
    Group records that are exact duplicates on specified columns.

    dataframe : the data frame to be split
    columns : columns along which to look for duplicate values
    group_label : the name to assign to the column containing duplicate values
    tag : an optional tag to append to group assigments


    '''

    if type(columns) == str:
        columns = [columns]

    if type(dataframe) == list:
        df = dataframe[0].copy()
    else:
        df = dataframe.copy()

    topdown = df[columns].apply(
        lambda x: x.duplicated(take_last=False))
    bottomup = df[columns].apply(
        lambda x: x.duplicated(take_last=True))
    exact_matches = topdown | bottomup

    duplicate_index = exact_matches.apply(np.sum, axis=1) > 0

    if duplicate_index.sum() > 0:
        duplicates = df[duplicate_index]
        not_duplicates = df[~duplicate_index]

        replacements = pd.Series(
            duplicates.groupby(columns).grouper.group_info[0],
            index=duplicates.index)

        if tag is not None:
            replacements = replacements.apply(str) + tag

        duplicates[group_label] = replacements

        df = pd.concat([duplicates, not_duplicates])
    else:
        df[group_label] = None

    return df


def dataframe_linkage(records, settings, blocking, id_col=None, to_string=False,
    reset_index=True, threshold=None, keep_best=True, override_columns=None, 
    exclude_exact_matches=True, bootstrap=None, verbose=True):
    '''
    records : data frame or list of data frames to convert to dictionaries;
        columns to be used in records linkage must be consistently named across
        data frames; if a list or dictionary or two dataframes is passed to
        records, the two dataframes will be linked. Otherwise, the function
        assumes a single dataframe and will label duplicates
    settings : file or dictionary containing settings to pass to dedupe module
    blocking : column or list of columns on which to block records
    id_col: name of column containing unique identifiers (passed to
        `dataframe_to_frozendict`); id_col, like all columns, must be consistent
        across data frames
    to_string : boolean, indicating whether to convert all values to type str;
        passed to `dataframe_to_frozendict`
    reset_index : reset index of data frames as a precaution to make sure each
        row has a unique identifier
    threshold : float between 0.0 and 1.0 indicating cutoff point in logistic
        regression predictons for determining matches. If None, threshold will
        be determined algorithmically.
    keep_best : boolean, indicating whether to keep only the best potential match
        between two records
    override_columns : a columns of list of columns that automatically trigger
        a match; if two records match on all override_columns, they are labelled
        as a match and are not processed through the dedupe algorithms
    exclude_exact_matches : boolean, indicating whether to leave exact matches
        from override_columns out of the records for fuzzy matching
    bootstrap : integer, indicating how many samples to use in bootstrapping
        mean similarity
    verbose : boolean, indicating whether to output informative messages during
        the deduping process


    '''
    
    settings = {keys:(values if (type(values) is list) else [values]) for 
        (keys, values) in settings.items()}     
    
    if type(blocking) is not list:
        blocking = [blocking]
        
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
    
    if override_columns is not None:
    
        records = pd.concat(records, keys=input_ids)
    
        records = group_by_exact_matches(dataframe=records,
            columns=override_columns, group_label='cluster_id', tag='_exact')
    
        if exclude_exact_matches:
            exact_dupes = records[records['cluster_id'].notnull()]
            records = records[records['cluster_id'].isnull()]
    
        if len(input_ids) == 2:
            records = [records[records.index.get_level_values(0) == x].copy()
                for x in input_ids]
            exact_dupes = [exact_dupes[exact_dupes.index.get_level_values(0) == x].copy()
                for x in input_ids]
            exact_dupes[0] = (pd.merge(exact_dupes[0], 
                exact_dupes[1].loc[:,['cluster_id', 'id']], how='left', 
                on='cluster_id', suffixes=('', '_match')))
            exact_dupes[1] = (pd.merge(exact_dupes[1], 
                exact_dupes[0].loc[:,['cluster_id', 'id']], how='left', 
                on='cluster_id', suffixes=('', '_match')))

        else:
            records = [records]
            exact_dupes = [exact_dupes]
        
    for i in range(len(records)):
        records[i] = records[i].drop(['cluster_id'], axis=1)
        blocker = records[i][blocking]
        records[i].columns = records[i].columns.values + input_ids[i]

    
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
            wanted_cols = [x for x in temp_records.columns if x.startswith(col)]
            pairs = temp_records[wanted_cols].to_records(index=False)
            for fun in settings[col]:
                colname = col + '__' + fun.__name__
                results[colname] = fun(pairs)
            if verbose:
                print '%d ' % (target_columns.index(col) + 1),
        
        if bootstrap is None:
            means = results.set_index(id_columns).mean(axis=1).reset_index(
                drop=True)
        elif bootstrap == 'jackknife':
            boot_df = results.set_index(id_columns)
            means = []
            all_cols = boot_df.columns.values
            
            for col in all_cols:
                jack_cols = [x for x in all_cols if x != col]
                jack_means = boot_df[jack_cols].mean(axis=1).reset_index(drop=True)
                means.append(jack_means)
            means = pd.concat(means, axis=1).mean(axis=1)
        else:
            boot_df = results.set_index(id_columns)
            means = []
            all_cols = boot_df.columns.values
            n_cols = len(all_cols)            

            for i in range(bootstrap):
                boot_cols = np.random.choice(all_cols, size=n_cols, replace=True)
                boot_means = boot_df[boot_cols].mean(axis=1).reset_index(drop=True)
                means.append(boot_means)
            means = pd.concat(means, axis=1).mean(axis=1)
        
        results['mean_similarity'] = means

        if threshold is not None:        
            
            keeps = (means > threshold).values
            results = results[keeps]
        
        if keep_best:
           results = results.sort(columns='mean_similarity', ascending=False)
           results = results.groupby(id_columns[0]).first().reset_index()
           results = results.sort(columns='mean_similarity', ascending=False)
           results = results.groupby(id_columns[1]).first().reset_index()

        matched_df = matched_df.append(results, ignore_index=True)
        print ''

    matched_df['cluster_id'] = range(matched_df.shape[0])
    keep_columns = id_columns + ['cluster_id', 'mean_similarity']
    dupes = matched_df[keep_columns]

    records[0] = pd.merge(records[0], dupes, how='left', on=id_columns[0])
    records[1] = pd.merge(records[1], dupes, how='left', on=id_columns[1])
    
    records[0] = records[0].rename(columns={id_columns[0]: id_col, 
        id_columns[1]: id_col+'_match'})
    records[1] = records[1].rename(columns={id_columns[1]: id_col, 
        id_columns[0]: id_col+'_match'})

    records[0].columns = [x.replace(input_ids[0], '') for x in 
        records[0].columns]
    records[1].columns = [x.replace(input_ids[1], '') for x in 
        records[1].columns]

    if (override_columns is not None) & exclude_exact_matches:
        records[0] = pd.concat([records[0], exact_dupes[0]], ignore_index=True)
        records[1] = pd.concat([records[1], exact_dupes[1]], ignore_index=True)
    
    start_cols = [id_col, id_col+'_match', 'cluster_id', 'blocker', 'mean_similarity']
    other_cols = [x for x in records[0].columns if x not in start_cols]    
    column_order = start_cols + other_cols    
    records = [x[column_order] for x in records]
    
    if record_type == dict:
        records = {input_ids[i].replace('_', ''):records[i] for i in 
            range(len(input_ids))}
    
    return records

def get_pairs(df1, df2, id_col, match_col, comparison_cols, metric_col=None,
              how='inner'):

    if metric_col is None:
        metric_col = []
    else:
        metric_col = [metric_col]

    df1_nomatch = df1[df1[match_col].isnull()]
    df2_nomatch = df2[df2[match_col].isnull()]
        
    df1 = df1[df1[match_col].notnull()]
    df2 = df2[df2[match_col].notnull()]
    
    df1 = df1.sort(columns=match_col)
    df2 = df2.sort(columns=id_col)
    
    df1 = df1[[match_col] + comparison_cols]
    df2 = df2[[id_col, match_col] + comparison_cols + metric_col]

    df1_nomatch = df1_nomatch[[match_col] + comparison_cols]
    df2_nomatch = df2_nomatch[[id_col, match_col] + comparison_cols + metric_col]

    df1 = pd.concat([df1, df1_nomatch], ignore_index=True)
    df2 = pd.concat([df2, df2_nomatch], ignore_index=True)
    
    output = pd.merge(df1, df2, how=how, left_on=match_col, right_on=id_col, 
        suffixes=('_2', '_1'))

    output = output.drop([id_col], axis=1)

    return output


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
