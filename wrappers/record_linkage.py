#!/usr/bin/python
# -*- coding: utf-8 -*-

import os
import pandas as pd
import numpy as np
import dedupe

def identical_comparator(field_1, field_2) :
    if field_1 and field_2 :
        if field_1 == field_2 :
            return 1
        else:
            return 0
    else :
        return np.nan

def absolute_difference_comparator(field_1, field_2):
    abs_diff = np.abs(field_1 - field_2)
    return abs_diff

def log1p_difference_comparator(field_1, field_2):
    log1p_diff = np.log1p(abs(field_1 - field_2))
    return log1p_diff

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


def dataframe_linkage(records, settings, id_col=None, to_string=False,
    reset_index=True, records_sample=0.01, training_file=None, training='append',
    settings_output=None, threshold=None, recall_weight=1.5,
    split_output=False, override_columns=None, exclude_exact_matches=True,
    ppc=1, uncovered_dupes=1, verbose=True):
    '''
    records : data frame or list of data frames to convert to dictionaries;
        columns to be used in records linkage must be consistently named across
        data frames; if a list or dictionary or two dataframes is passed to
        records, the two dataframes will be linked. Otherwise, the function
        assumes a single dataframe and will label duplicates
    settings : file or dictionary containing settings to pass to dedupe module
    id_col: name of column containing unique identifiers (passed to
        `dataframe_to_frozendict`); id_col, like all columns, must be consistent
        across data frames
    to_string : boolean, indicating whether to convert all values to type str;
        passed to `dataframe_to_frozendict`
    reset_index : reset index of data frames as a precaution to make sure each
        row has a unique identifier
    records_sample : number of random pairs to pull from data for training; if less
        than 1, sample size will equal sample size times the number of possible
        combinations of all rows across data frames
    training_file : if not None, the location of of a json file containing already-
        categorized pairs
    training : 'replace', 'append', or 'none' indicating whether to replace any
        existing training files with new training data, to append new training
        data, or to skip training altogether
    block_sample : float betwen 0.0 and 1.0, indicating the percentage of blocked
        data to sample in determining a good threshold for clustering
    recall_weight : weight to determin threshold to maximize a weighted average of
        precision and recall (recall_weight==2 means we care twice as much about
        recall as we do precision)
    threshold : float between 0.0 and 1.0 indicating cutoff point in logistic
        regression predictons for determining matches. If None, threshold will
        be determined algorithmically.
    split_output : boolean, indicating whether to return a list of data frames
        corresponding to the structure of the input list; if split_output==False,
        a single data frame will be returned with a 'data_frame_id' column
        indicating which records belonged to which item of the input list; if
        only one data frame is input, this parameter will be ignored
    override_columns : a columns of list of columns that automatically trigger
        a match; if two records match on all override_columns, they are labelled
        as a match and are not processed through the dedupe algorithms
    exclude_exact_matches : boolean, indicating whether to leave exact matches
        from override_columns out of the records for fuzzy matching
    ppc : float between 0.0 and 1.0; the proportion of all possible pairs a 
        predicate is allowed to cover. If a predicate puts together a fraction 
        of possible pairs greater than ppc, it will be removed from consideration. 
        Passed to dedupes `blockingFuncton`
    uncovered_dupes : integer, the number of true dupes pairs in the training
        that can fail to be placed in a block. Passed to dedupes `blockingFunction`
    verbose : boolean, indicating whether to output informative messages during
        the deduping process


    '''

    if type(settings) is dict:
        settings = dict(settings)

    if training_file is not None:
        if (not os.path.exists(training_file)) & (training == 'none'):
            raise Exception('must supply training_file or do active training')
        elif os.path.exists(training_file) & (training == 'replace'):
            os.remove(training_file)

    record_type = type(records)

    # Convert all data frames to frozen dictionaries
    if verbose:
        print 'preparing records for linkage...'
    
    if record_type == dict:
        input_ids = [(x+'_') for x in records.keys()]
        records = [x for x in records.values()]
    elif record_type == list:
        input_ids = ['first_', 'second_']
    elif record_type == pd.DataFrame:
        input_ids = [None]
        records = [records]
        

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
        else:
            records = [records]
            exact_dupes = [exact_dupes]

    if id_col is not None:
        records = [x.set_index(id_col, drop=False) for x in records]
    elif reset_index:
        records = [x.reset_index(drop=True) for x in records]

    for i in range(len(records)):
        records[i].index = [str(x) for x in records[i].index]

    record_dicts = []
    for i in range(len(records)):
        record_d = dataframe_to_recorddict(records[i], id_col=id_col,
            df_tag = input_ids[i], to_string=to_string)
        record_dicts.append(record_d)

    if records_sample < 1:
        n_dicts = len(record_dicts)

        if n_dicts == 1:
            n_records = sum([len(x) for x in record_dicts])
            n_combinations = n_records * (n_records - 1) / 2
        else:
            n_combinations = np.product([len(x) for x in record_dicts])

        records_sample = int(np.ceil(n_combinations * records_sample))
    
    # Instantiate appropriate Dedupe class
    if len(records) == 2:
        linker = dedupe.api.RecordLink(settings)
    else:
        linker = dedupe.Dedupe(settings)

    if os.path.exists(training_file):
        if verbose:
            print 'reading in existing training file...'
        linker.readTraining(training_file)
    
    if type(settings) is dict:
        
        if training != 'none':
            # Sample data for active learning
    
            if verbose:
                print 'randomly sampling records to perform active labelling...'
        
            if len(records) == 2:
                linker.sample(record_dicts[0], record_dicts[1], records_sample)
            else:
                linker.sample(record_dicts[0], records_sample)
        
            if verbose:
                print 'initiating active labeling...'
            
            dedupe.consoleLabel(linker)
    
        if verbose:
            print 'training model...'
        
        linker.train(ppc=ppc, uncovered_dupes=uncovered_dupes)
    
        if training_file is not None:
            linker.writeTraining(training_file)
        if settings_output is not None:
            linker.writeSettings(settings_output)
    
        if threshold == None:
            if verbose:
                print 'blocking...'
            if len(records) == 2:
                threshold = linker.threshold(record_dicts[0], record_dicts[1], 
                    recall_weight=recall_weight)
            else:
                threshold = linker.threshold(record_dicts[0], 
                    recall_weight=recall_weight)
    
    if verbose:
        print 'clustering...'

    if len(records) == 2:
        clustered_dupes = linker.match(record_dicts[0], record_dicts[1], 
            threshold)
    else:
        clustered_dupes = linker.match(record_dicts[0], threshold)

    if verbose:
        print '# duplicate sets', len(clustered_dupes)

    if len(records)==2:
        clustered_dupes = enumerate(clustered_dupes)
        clustered_dupes = [(y, z, x) for (x,(y,z))  in clustered_dupes]
        record_0_dupes = pd.Series({ str(x).replace(input_ids[0], ''):z for 
            (x,y,z) in clustered_dupes })
        record_1_dupes = pd.Series({ str(y).replace(input_ids[1], ''):z for 
            (x,y,z) in clustered_dupes })
        record_0_dupes = record_0_dupes.reindex(index=records[0].index)
        record_1_dupes = record_1_dupes.reindex(index=records[1].index)

        records[0]['cluster_id'] = record_0_dupes
        records[1]['cluster_id'] = record_1_dupes
        
        if (override_columns is not None) & exclude_exact_matches:
            records[0] = pd.concat([records[0], exact_dupes[0]], ignore_index=True)
            records[1] = pd.concat([records[1], exact_dupes[1]], ignore_index=True)
    
    else:
        # create a 'cluster_id' column to show which records fall within the same cluster
        clustered_dupes = enumerate(clustered_dupes)
        clustered_dupes = [zip(y, [x]*len(y)) for (x, y) in clustered_dupes]
        clustered_dupes = [x for y in clustered_dupes for x in y]
        cluster_membership = pd.Series({ x:y for (x,y) in clustered_dupes })
        cluster_membership.index = [str(x) for x in cluster_membership.index]
        cluster_membership = cluster_membership.reindex(index=records[0].index)

        records['cluster_id'] = cluster_membership

        if (override_columns is not None) & exclude_exact_matches:
            records[0] = pd.concat([records[0], exact_dupes[0]], ignore_index=True)
    
    if record_type == dict:
        records = {input_ids[i].replace('_', ''):records[i] for i in range(len(input_ids))}
    elif record_type == pd.DataFrame:
        records = records[0]

    return records
