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

# Function to convert data frame to hashable dictonary
def dataframe_to_frozendict(df, id_col, to_string=False):
    '''
    df : data frame to convert to a frozen dictionary
    id_col : column of unique identifiers; if None (default), the data frame
        index will be used
    to_string : boolean, indicating whether to convert all values to type str

    '''

    if id_col is not None:
        id_values = df[id_col].values
    else:
        id_values = df.reset_index().index.values

    colnames = df.columns.values

    df = df.fillna('')

    if to_string:
        df = df.applymap(str)

    df= df.groupby(id_values)
    df = df.apply(lambda x: dedupe.core.frozendict(zip(colnames, x.values[0])))
    df.index = [int(x) for x in df.index.values]

    df_d = df.to_dict()

    return df_d


def dataframe_linkage(records, settings, id_col=None, to_string=False,
    records_sample=0.01, training_file=None, training='append',
    settings_output=None, threshold_sample=1.0, recall_weight=1.5,
    split_output=False, verbose=True):
    '''
    records : data frame or list of data frames to convert to a frozen dictionary;
        columns to be used in records linkage must be consistently named across
        data frames
    settings : file or dictionary containing settings to pass to dedupe module
    id_col: name of column containing unique identifiers (passed to
        `dataframe_to_frozendict`); id_col, like all columns, must be consistent
        across data frames
    to_string : boolean, indicating whether to convert all values to type str;
        passed to `dataframe_to_frozendict`
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
    split_output : boolean, indicating whether to return a list of data frames
        corresponding to the structure of the input list; if split_output==False,
        a single data frame will be returned with a 'data_frame_id' column
        indicating which records belonged to which item of the input list; if
        only one data frame is input, this parameter will be ignored
    verbose : boolean, indicating whether to output informative messages during
        the deduping process


    '''

    # Instantiate Dedupe class
    deduper = dedupe.Dedupe(settings)

    # Convert all data frames to frozen dictionaries
    record_type = type(records)
    if record_type == dict:
        input_ids = records.keys()
    elif record_type == list:
        input_ids = range(len(records))
    else:
        input_ids = 'single data frame'
    for i in input_ids:
        records[i]['data_frame_id'] = input_ids[i]

    records = pd.concat(records, ignore_index=True).set_index(id_col, drop=False)

    record_dicts = dataframe_to_frozendict(records, id_col=id_col,
        to_string=to_string)

    if type(settings) == dict:

        if records_sample < 1:

            def factorial(number):
                product=1
                for i in range(number):
                    product = product * (i+1)
                return product

            n=len(record_dicts)
            n_combinations = factorial(n + 2 - 1)/(factorial(2)*factorial(n-1))
            records_sample = int(np.ceil(n_combinations * records_sample))

        # Set up training data/files
        if training != 'none':
            if os.path.exists(training_file) & (training == 'append'):
                deduper._initializeTraining(training_file)

            data_sample = dedupe.dataSample(record_dicts, records_sample)

            if verbose:
                print 'perform active labeling...'
            deduper.train(data_sample, dedupe.training.consoleLabel)
        elif os.path.exists(training_file):
            deduper.train(data_sample, training_file)
        else:
            Exception('no training information (file or active labelling) provided')

        if training_file is not None:
            deduper.writeTraining(training_file)

    if verbose:
        print 'blocking...'
    blocker = deduper.blockingFunction()

    if settings_output is not None:
        deduper.writeSettings(settings_output)

    # Load all the original data in to memory and place them in to blocks.
    blocked_data = dedupe.blockData(record_dicts, blocker)

    # Clustering
    if threshold_sample < 1:
        n_blocks = len(blocked_data)
        block_sample = np.random.choice(n_blocks, n_blocks * threshold_sample,
            replace=True)
        threshold = deduper.goodThreshold(
            tuple([blocked_data[x] for x in block_sample]),
            recall_weight=recall_weight)
    else:
        threshold = deduper.goodThreshold(blocked_data,
            recall_weight=recall_weight)

    if verbose:
        print 'clustering...'
    clustered_dupes = deduper.duplicateClusters(blocked_data, threshold)

    if verbose:
        print '# duplicate sets', len(clustered_dupes)

    # create a 'cluster_id' column to show which records fall within the same cluster
    clustered_dupes = [zip(y, [x]*len(y)) for (x, y) in
        enumerate(clustered_dupes)]
    clustered_dupes = [x for y in clustered_dupes for x in y]
    cluster_membership = pd.Series({ x:y for (x,y) in clustered_dupes })
    cluster_membership.index = [int(x) for x in cluster_membership.index]
    cluster_membership = cluster_membership.reindex(
        index=[int(x) for x in records.index])
    cluster_membership.index = records.index

    records['cluster_id'] = cluster_membership

    if split_output == True:
        if record_type == dict:
            records = {x:records[records['data_frame_id'] == x] for x in input_ids}
        elif record_type == list:
            records = [records[records['data_frame_id'] == x] for x in input_ids]
        else:
            print 'only one data frame was input, so records cannot be split'

    return records
