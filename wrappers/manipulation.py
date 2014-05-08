# -*_ coding: utf-8 -*-
"""
Created on Fri Mar 28 16:58 2014

@authors:
    schaunwheeler
    paulmeinshausen
"""

import pandas as pd
import sklearn.feature_extraction.text as skfet
import nltk.stem.porter

def stem_sentence(sentence):
     stemmer = nltk.stem.porter.PorterStemmer()
     out = [stemmer.stem(x) for x in sentence.split(' ')]
     return ' '.join(out)

def pandas_ngram(series, nrange = 1, min_df = 1, max_df = 1.0,
    max_features = None, missing_column = True, clean = '[^A-Za-z0-9 ]',
    remove_correlates = 1.0, stem=True):

    '''
    Takes a Pandas series and transforms it into a data frame
    of ngram counts.

    Parameters
    ----------
    series:          a pandas series of text strings\n
    nrange:          a list of numbers indicating which ngrams to identify.
                     An integer will create min(nrange)-grams through
                     max(nrange)-grams\n
    min_df,
    max_df,
    max_features:    passed to sklearn\n
    missing_column:  boolean, indicating whether to createa a column to
                     identify records where no n-grams were found\n
    clean:           a regular expression to use to clean the data before
                     vectorizing\n
    remove_correlates:  float, giving the threshold for removing correlated
                        ngram counts. If two vectors have a pearson correlation
                        greater than or equal to remove_correlates, only the
                        longest ngram will be kept.\n

    '''

    if clean is not None:
        input_series = series.str.replace(clean, '')
    else:
        input_series = series

    if stem:
        input_series = input_series.apply(stem_sentence)

    if type(nrange) == int:
        nrange = range(nrange)
        nrange = [x + 1 for x in nrange]
    nrange = sorted(nrange, reverse = True)

    output = pd.DataFrame()

    for n in nrange:
        counter = skfet.CountVectorizer(analyzer = 'word',
                                        ngram_range = (n, n),
                                        stop_words = 'english',
                                        lowercase = True,
                                        min_df = min_df,
                                        max_df = max_df,
                                        max_features = max_features)

        counts = counter.fit_transform(input_series)

        count_df = pd.DataFrame(counts.toarray(),
                                columns = counter.get_feature_names(),
                                index = series.index)

        if missing_column:
            has_nonzero = count_df.abs().sum(axis = 1) > 0
            if has_nonzero.sum() > 0:
                count_df['missing_%sgram' % n] = ~has_nonzero * 1

        output = pd.concat([output, count_df], axis = 1)

    if remove_correlates is not None:
        remove_list = []
        output_corr = output.corr().abs() >= remove_correlates
        output_corr = output_corr.loc[output_corr.sum() > 1,
                                      output_corr.sum() > 1]

        for ngram in output_corr.index:
            candidates = output_corr[ngram][output_corr[ngram]].index
            candidates = sorted(candidates, key = len)
            keep = candidates.pop()
            remove_list.append(candidates)
        remove_list = [x for y in remove_list for x in y]
        remove_list = list(set(remove_list))
        if len(remove_list) > 1:
            output = output.drop(remove_list, axis = 1)

    return output

def create_dummies(data, column_dict , threshold=None, drop_original=True,
                   seperator='___'):

    dummy_cols = []
    for col in column_dict.keys():
        regex = column_dict[col]
        if regex != '':
            replace = '_'
        else:
            replace = ''

        dummy_out = pd.get_dummies(
                data[col].str.lower().str.replace(regex, replace),
                prefix=col, prefix_sep=seperator)

        if threshold is not None:
            dummy_keep = dummy_out.mean()
            dummy_keep = dummy_out.columns[dummy_keep>=threshold]
            dummy_out = dummy_out[dummy_keep]
            dummy_out[col+seperator+'others'] = (dummy_out.sum(axis=1)==0) * 1

        dummy_cols += dummy_out.columns.values.tolist()

        if drop_original:
            data = data.drop([col], axis=1)

        data = pd.concat([data, dummy_out], axis=1)

    return data


