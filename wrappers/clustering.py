# -*- coding: utf-8 -*-

import pandas as pd
import numpy as np
import scipy
import scipy.spatial
import scipy.cluster
import fastcluster

def choose_nclusters(linkage_output, corr_df, nclust, weight_average=True,
                     verbose=True):
    '''
    This function iterates through different numbers of clusters to determine
    which number of clusters results in the greatest within-cluster homogeneity.

    Parameters
    ----------

    linkage_output : the output of fastcluster.linkage or scipy.cluster.hierarchical.linkage
    corr_df : a pandas data frame of the correlation matrix of the variables used in the
        clustering
    nclust : the maximum number of clusters to consider. The function will estimate
        within-cluster homogeniety for 1 cluster up through nclust clusters.
    weight_average : boolean, indicating whether the average within-cluster correlation
        should be weighted by the number of variables included in each cluster

    '''

    weighted_average = pd.DataFrame(columns=['mean_internal', 'min_internal',
            'max_internal', 'mean_external', 'min_external', 'max_external'])
    corr_df.values[np.diag_indices_from(corr_df.values)] = np.nan

    for i in [x+1 for x in range(nclust)]:
        if verbose:
            print i,
        cluster_assignments = scipy.cluster.hierarchy.fcluster(
            linkage_output, i, 'maxclust')

        number_clusters = np.unique(cluster_assignments)

        summaries = pd.DataFrame(columns=['mean_internal', 'min_internal',
            'max_internal', 'mean_external', 'min_external', 'max_external'])

        for cluster in number_clusters:
            mask = cluster_assignments==cluster
            corr_mat_int = corr_df.ix[mask,mask]
            corr_mat_ext = corr_df.ix[~mask,mask]
            int_summary = corr_mat_int.describe().T[['mean', 'min', 'max']]
            int_summary = int_summary.fillna(1)
            ext_summary = corr_mat_ext.describe().T[['mean', 'min', 'max']]
            ext_summary.values[np.isnan(ext_summary.values)] = 0
            int_summary.columns = int_summary.columns + '_internal'
            ext_summary.columns = ext_summary.columns + '_external'

            int_ext = pd.concat([int_summary, ext_summary], axis=1)

            if weight_average:
                int_ext['weights'] = corr_mat_int.describe().T[['count']] + 1
            else:
                int_ext['weights'] = 1

            summaries = summaries.append(int_ext)

        summaries = summaries.drop(['weights'], axis=1).apply(np.average,
            weights=summaries['weights'].values)

        weighted_average = weighted_average.append(summaries.to_frame().T,
            ignore_index=True)

    weighted_average.index = weighted_average.index + 1
    weighted_average.index.name = 'n_clusters'

    return weighted_average
