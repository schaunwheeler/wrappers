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
from sklearn.decomposition import KernelPCA
from sklearn.cluster import AffinityPropagation


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
            corrs *= jaccard
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
        self.assignment_dict = pd.Series(self.cluster_assignments, index=self.correlation_matrix.index).to_dict()

    def dendrogram(self, plot=True, **kwargs):
        no_plot = not plot
        dend = dendrogram(
            self.linkage_output, labels=self.correlation_matrix.columns, color_threshold=self.threshold,
            no_plot=no_plot, **kwargs)
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
                if len(c_corr) == 0:
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

    def group_recursive(
            self, corr_method='pearson', weight=True, absolute=False, auto_exclude=None, linkage_method='complete',
            linkage_kwargs=None, t=0.0525, criterion='distance', scale=False, transform=None, **fclust_kwargs):
        linkage_kwargs = {} if linkage_kwargs is None else linkage_kwargs
        self.corr_dist(
            method=corr_method, weight=weight, absolute=absolute, auto_exclude=auto_exclude, scale=scale,
            transform=transform)
        self.linkage(method=linkage_method, **linkage_kwargs)
        self.assign_clusters(t=t, criterion=criterion, **fclust_kwargs)
        keep = [] if auto_exclude is None else auto_exclude

        i = 0
        old_tag = 'cluster_id'
        tag = 'cluster_id_'+str(i)
        n_clusts = np.inf
        new_data = self.append_cluster_col(tag=tag)
        n_clusts_new = new_data[tag].nunique()
        reserve = new_data.copy()
        reserve['ind'] = new_data.index
        
        while n_clusts_new < n_clusts:
            reserve_cols = [x for x in self.cluster_cols if x not in keep+[old_tag]]
            new_cluster_vars = [tag] + keep
            new_index_vars = [x for x in self.index_cols if x not in reserve_cols + new_cluster_vars + [old_tag]]
    
            reserve['ind'] = reserve[tag]
            new_data = new_data.groupby(new_index_vars+new_cluster_vars)
            new_data = new_data[self.value_var].sum().reset_index()
    
            self.__init__(
                data=new_data, value_var=self.value_var, cluster_vars=new_cluster_vars,index_vars=new_index_vars)
            self.corr_dist(
                method=corr_method, weight=weight, absolute=absolute, auto_exclude=auto_exclude, scale=scale,
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
            
            self.__init__(data=new_data, value_var=self.value_var, cluster_vars=keep+[tag], index_vars=new_index_vars)
            
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
        
    def quick_run(
            self, corr_method='pearson', weight=True, auto_exclude=None, absolute=False, link_method='complete',
            threshold=0.0525, scale=False, criterion='distance', transform=None, plot=True):
        self.corr_dist(
            method=corr_method, weight=weight, absolute=absolute, auto_exclude=auto_exclude, scale=scale,
            transform=transform)
        self.linkage(method=link_method)
        self.assign_clusters(t=threshold, criterion=criterion)
        self.dendrogram(plot=plot)
        self.summarize()
        return self.summary


def bulk_linear_impute(y, x=None, poly=None, log=False):
    """Impute missing values for an entire data frame using sklearn.linear_model.LinearRegression

    Args:
        y (pandas.DataFrame): a dataframe where each row is a time period and each column is a metric to be predicted.
        x (numpy.array, optional): time periods of length y.shape[0]; default (None), assumes to be the index of y
        poly (list): polynomials to build into the prediction
        log (boolean): whether to log y before predicting

    Returns:
        pd.DataFrame: original data frame with missing values filled in


    """


    nulls = y.isnull().multiply(y.index.values, axis=0).mean()
    null_vals = nulls[nulls > 0.].unique()

    for null_val in null_vals:
        use = (nulls == null_val).values
        y_use = y.loc[:, use].copy()
        if x is not None:
            x_use = x.copy()
        else:
            x_use = y.index.values
        if log:
            y_use = np.log(y_use+1)
        to_test = (y_use.isnull().sum(axis=1) > 0.0).values
        train = pd.DataFrame({'x': x_use[~to_test]})
        test = pd.DataFrame({'x': x_use[to_test]})
        if poly is not None:
            for p in poly:
                train['x' + str(p)] = train['x'] ** p
                test['x' + str(p)] = test['x'] ** p
        f = LinearRegression().fit(X=train, y=y_use.dropna())
        out = f.predict(test)[0]
        if log:
            out = np.exp(out) - 1
        y.loc[to_test, use] = out

    return y


def bulk_lag_estimate(y, shift, x=None, poly=None, log=False):
    """Create estimates for a specific time horizon for an entire data frame.

    No missing values permitted.

    Args:
        y (pandas.DataFrame): a dataframe where each row is a time period and each column is a metric to be predicted.
        shift (int): number of time periods to predict ahead
        x (numpy.array, optional): time periods of length y.shape[0]; default (None) assumes to be the index of y
        poly (list): polynomials to build into the prediction
        log (boolean): whether to log y before predicting

    Returns:
        pd.Series: series of estimates for horizon specified by `shift`


    """

    y_use = y.copy()
    if x is not None:
        x_use = x.copy()
    else:
        x_use = y.index.values
    if log:
        y_use = np.log(y_use+1)
    train = pd.DataFrame({'x': x_use})
    target = x_use.max()+shift
    test = pd.DataFrame({'x': [target]})
    if poly is not None:
        for p in poly:
            train['x' + str(p)] = train['x'] ** p
            test['x' + str(p)] = test['x'] ** p
    f = LinearRegression().fit(X=train, y=y_use)
    out = f.predict(test)[0]
    if log:
        out = np.exp(out) - 1

    out = pd.Series(out, index=y.columns)
    out.name = target

    return out


def make_weights(pool, q):
    pool = np.array(pool)
    probs = np.argsort(pool)
    probs = probs / probs.sum()
    probs = 1.0 / (1.0 + (np.abs(q - probs)))
    np.place(probs, np.isinf(probs) | np.isnan(probs), 0)
    probs = probs / probs.sum()
    return probs


class MarketSizes(object):
    """Class to load and manipulate Euromonitor Market Sizes data.

    This class's methods make a lot of assumptions about the structure of the market sizes table structure.



    """

    def __init__(self, conn):
        """Establish database connection and load some data for later use

        Args:
            conn (sqlalchemy.engine.base.Engine): a database engine created through SQLAlchemy

        Attributes:
            countries_ (pandas.DataFrame): country id (iso3) and name
            categories_ (pandas.DataFrame): Euromonitor product category id and name
            country_dict (dict): dictionary to map country ids to names
            category_dict (dict): dictionary to map category ids to names
            data_ (pandas.DataFrame): placeholder for data
            lagged (dict): container for lagged data sets
            extrapolated (dict): container for extrapolated data sets
            staging_data_ (pandas.DataFrame): lagged and reshaped data, ready for analysis


        """
        self.conn = conn
        self.countries_ = pd.read_sql('SELECT iso3 AS id, name FROM countries_country', conn)
        self.country_dict = self.countries_.set_index('id')['name'].to_dict()
        self.categories_ = pd.read_sql('SELECT id, name FROM industries_emcategory', conn)
        self.category_dict = self.categories_.set_index('id')['name'].to_dict()
        self.data_ = pd.DataFrame()
        self.lagged = {}
        self.extrapolated = {}
        self.staging_data_ = pd.DataFrame()

    def get_data(self, years=None, forecasts=False, silent=True):
        """Load Euromonitor Market Sizes data and some other metrics.

        Args:
            years (list): list of years to pull
            forecasts (boolean): whether to includes years where data is a Euromonitor estimate
            silent (boolean): whether to return dataframe containing data

        Returns:
            nothing if silent==True. Data will be put in data_ attribute.
            Data has following columns:
                country_id: iso3 code
                year: year
                category_id: euromonitor category id
                type: category type (on-trade, off-trade, foodservice, etc.)
                gdp_ppp: GDP at purchasing power parity (from WorldBank)
                gdp_percapita: GDP per capita (from WorldBank)
                spend_total: total constant 2011 dollars spent on category within country/year combination
                population: total population (from UN)
                spend_percapita: spend total divided by population
                spend_share: spend_total divided by sum of spend total of top-level categories after omiting
                    on-trade/off-trade distinctions


        """

        # format years for inclusion in SQL statement
        years = range(1950, 2050) if years is None else years
        years = ', '.join([str(x) for x in years])

        # exclude these countries (regions) from pull
        countries = ['AA', 'AP', 'EE', 'LA', 'MA', 'NAC', 'WE', 'WLD']
        countries = ', '.join([repr(x) for x in countries])

        # format WorldBank variables for inclusion in SQL statement
        variables = ['NY_GDP_PCAP_PP_KD', 'NY_GDP_MKTP_PP_KD']
        variables = ', '.join([repr(x) for x in variables])

        marketsizes = pd.read_sql(
            '''
            SELECT
                ms.country_id,
                ms.year,
                ms.emcategory_id as category_id,
                type.name as type,
                ms.first_forecast,
                /*ms.type_id,*/
                /*ms.fx_id,*/
                /*ms.price_id,*/
                /*ms.unit_id,*/
                ms.size * (fx.value / fxbaseline.value) * 1000000 as spend_total
            FROM industries_marketsize as ms
            LEFT JOIN countries_countrydata AS fx
                ON ms.country_id = fx.country_id
                AND ms.year = fx.year
                AND fx.indicator_id = 'fx'
            LEFT JOIN countries_countrydata AS fxbaseline
                ON ms.country_id = fxbaseline.country_id
                AND fxbaseline.year = 2011
                AND fxbaseline.indicator_id = 'fx'
            LEFT JOIN industries_type AS type
                ON ms.type_id = type.id
            WHERE ms.size IS NOT NULL
                AND fx.value IS NOT NULL
                AND fxbaseline.value IS NOT NULL
                AND ms.fx_id=1
                AND ms.unit_id=10
                AND ms.type_id IN (3, 9, 10, 11, 12, 13)
                AND ms.country_id NOT IN ({c})
                AND ms.year IN ({y});
            '''.format(y=years, c=countries), self.conn)

        population = pd.read_sql(
            '''
            SELECT
                country_id,
                year,
                SUM(people)*1000 AS value
            FROM countries_population
            WHERE country_id NOT IN ({c})
                AND year IN ({y})
            GROUP BY country_id, year;
            '''.format(y=years, c=countries), self.conn)
        population['indicator_id'] = 'population'

        features = pd.read_sql(
            '''
            SELECT
              country_id,
              year,
              indicator_id,
              value
            FROM countries_countrydata
            WHERE country_id NOT IN ({c})
              AND indicator_id IN ({v})
              AND year IN ({y});
            '''.format(y=years, c=countries, v=variables), self.conn)
        features = pd.concat([features, population])
        features = features.set_index(['country_id', 'year', 'indicator_id'])
        features = features['value'].unstack(['country_id', 'indicator_id'])

        # Impute missing values for WorldBank indicators
        features1 = bulk_linear_impute(features, log=True)
        features2 = bulk_linear_impute(features, poly=[2], log=True)
        features = (features1 + features2) / 2.0
        features = features.stack('country_id').reset_index()
        features = features.rename(columns={'NY_GDP_MKTP_PP_KD': 'gdp_ppp', 'NY_GDP_PCAP_PP_KD': 'gdp_percapita'})

        # Merge demographic data with marketsizes data
        data = pd.merge(marketsizes, features, how='left', on=['year', 'country_id'])

        # Calculate alternative consumption measures
        data['spend_percapita'] = data['spend_total'] / data['population']
        data = data.dropna(subset=['spend_total', 'spend_percapita'])

        def calc_spend_share(df):
            """Calculate spend share by omitting on/off-trade distinctions from top-level categories"""

            ontrade = df['type'].str.contains('^On-trade')
            offtrade = df['type'].str.contains('^Off-trade')
            firstlevel = df['category_id'].str.endswith('-00-00-00-00-00-00')
            share = df['spend_total'] / df[~ontrade & ~offtrade & firstlevel]['spend_total'].sum()
            share[share == np.inf] = 0.0
            return share

        grp = data.groupby(['country_id', 'year'], group_keys=False, sort=False, squeeze=True)
        data['spend_share'] = grp.apply(calc_spend_share)

        if not forecasts:
            keep = (data['year'] < data['first_forecast']) | data['first_forecast'].isnull()
            data = data[keep].drop(['first_forecast'], axis=1)

        self.data_ = data.copy()

        if not silent:
            return data.copy()

    def lag(self, index, metrics, time, shift, silent=True):
        """Align lagged data with dataframe.

        Args:
            index (list): columns of self.data_ to use to form a unique index
            metrics (list): columns of self.data_ for which to create lagged versions
            time (list): columns of self.data_ that delineate time periods
            shift (int): number of time periods to lag
            silent (boolean): whether to silently keep output as attribute of class, or to return the object

        Returns:
            pandas.DataFrame: self.data_ with lagged variables appended


        """

        df = self.data_.copy()
        side_columns = [x for x in df.columns if x not in (index + metrics)]
        df = df.set_index(index)
        reserve = df[side_columns]  # hold out for later
        df = df.drop(side_columns, axis=1)
        df = df.stack()
        df.index.names = ['metric' if x is None else x for x in df.index.names]
        df = df.unstack([x for x in df.index.names if x not in time])
        df = pd.concat([df, df.shift(shift)], keys=[0, shift], names=['lag', 'year'])
        df = df.stack([x for x in df.columns.names if x != 'metric']).unstack('lag')
        df.columns = df.columns.reorder_levels(['lag', 'metric'])
        df = df.T.sort_index().T
        df.index = df.index.reorder_levels(index)
        reserve.index = reserve.index.reorder_levels(index)
        reserve.columns = [(0, x) for x in reserve.columns]
        dropna_cols = df.columns.values
        df = pd.concat([df, reserve], axis=1)
        df = df.dropna(subset=dropna_cols)
        df.columns = pd.MultiIndex.from_tuples(df.columns, names=['lag', 'metric'])
        self.lagged[shift] = df.copy()
        if not silent:
            return df.copy()

    def extrapolate(self, index, metrics, time, shift, silent=True):
        """Create simple linear predictions at a specified time horizon.

        Args:
            index (list): columns of self.data_ to use to form a unique index
            metrics (list): columns of self.data_ for which to create lagged versions
            time (list): columns of self.data_ that delineate time periods
            shift (int): number of time periods forward to forecast
            silent (boolean): whether to silently keep output as attribute of class, or to return the object

        Returns:
            pandas.DataFrame: self.data_ with extrapolations appended in new columns


        """

        df = self.data_.copy()
        side_columns = [x for x in df.columns if x not in (index + metrics)]
        df = df.set_index(index)
        reserve = df[side_columns]  # hold out for later
        df = df.drop(side_columns, axis=1)
        df = df.stack()
        df.index.names = ['metric' if x is None else x for x in df.index.names]
        df = df.unstack([x for x in df.index.names if x not in time])

        # impute missing values (extrapolation can't handle them)
        only_one = df.notnull().sum() == 1.0
        df.loc[:, only_one.values] = df.loc[:, only_one.values].replace(np.nan, 0.0)
        dfi1 = bulk_linear_impute(df, poly=[2], log=True)
        dfi2 = bulk_linear_impute(df, poly=None, log=True)
        df = (dfi1 + dfi2) / 2.0
        df[df < 0.0] = 0.0

        # extrapolate in any cases where we have at least three years
        df_years = df.index.values[2:]
        df_years = df_years[df_years <= (df.index.values.max() - shift)]
        df1 = [bulk_lag_estimate(df.loc[:i, :], shift, poly=[2], log=True) for i in df_years]
        df1 = pd.concat(df1, axis=1).T
        df2 = [bulk_lag_estimate(df.loc[:i, :], shift, poly=None, log=True) for i in df_years]
        df2 = pd.concat(df2, axis=1).T
        df_pred = (df1 + df2) / 2.0
        df_pred[df_pred < 0.0] = 0.0

        df = pd.concat([df, df_pred], keys=[0, shift], names=['forecast', 'year'])
        df = df.stack([x for x in df.columns.names if x != 'metric']).unstack('forecast')
        df.columns = df.columns.reorder_levels(['forecast', 'metric'])
        df = df.T.sort_index().T
        df.index = df.index.reorder_levels(index)
        reserve.index = reserve.index.reorder_levels(index)
        reserve.columns = [(0, x) for x in reserve.columns]
        dropna_cols = df.columns.values
        df = pd.concat([df, reserve], axis=1)
        df = df.dropna(subset=dropna_cols)
        df.columns = pd.MultiIndex.from_tuples(df.columns, names=['forecast', 'metric'])
        self.extrapolated[shift] = df.copy()

        if not silent:
            return df.copy()

    def reshape(self, lag, target, rows, predictors, category, threshold=0.1, silent=True):
        """
        Args:
            lag (str or int): key of output of calling `self.lag`; use a negative value to create test data
            target (str): column name of metric to populate values of new matrix
            rows (list): names of columns to use as rows in new matrix
            predictors (list): names of columns to use as predictors (columns) in new matrix
            category (dict): index names and values to subset data
            threshold (float): percentage of reshaped columns that must be non-null in order to keep row; rows with
                percentge of null values below threshold will have missing values filled with 0.0


        """

        lagger = lag if lag > 0 else 0
        df = self.lagged[lagger].copy() if lag > 0 else self.lagged.values()[0]
        pivot_columns = [x for x in df.index.names if x not in rows]
        df_target = df.copy()
        for key, val in category.items():
            df_target = df_target.xs(val, level=key, drop_level=False)
        df_target = df_target[[(0, target)]].unstack(pivot_columns)
        new_cols = [(-1,) + x[1:] for x in df_target.columns.values]
        df_target.columns = pd.MultiIndex.from_tuples(new_cols, names=df_target.columns.names)
        pred_columns = [(lagger, x) for x in predictors]
        df_match = df[pred_columns].unstack(pivot_columns)
        empties = (df_match.isnull().mean() == 1.0).values
        df_match = df_match.loc[:, ~empties]
        keep = df_match.notnull().mean(axis=1) >= threshold
        df_match = df_match.loc[keep, :]
        df_match = df_match.fillna(0.0)
        df = pd.merge(df_target, df_match, how='left', left_index=True, right_index=True)
        final_cols = [(np.abs(lag),) + x[1:] if x[0]!=-1 else x for x in df.columns.values]
        df.columns = pd.MultiIndex.from_tuples(final_cols, names=df.columns.names)
        self.staging_data_ = df.copy()

        if not silent:
            return df.copy()


class Ensemble(object):
    """Class to use ensemble (tree) methods to select features, cross-validate model fit, and predict to new data.


    """

    def __init__(self, data, yname, modeler, scale=True, seed=None, prep=False):
        """Load data and set up basic model parameters

        Args:
            data (pandas.DataFrame): the data to train the model
            yname (str): the name of the column representing the outcome variable
            prep (bool): whether to ensure the dataset has only float values in it
            scale (bool): whether to normalize the data
            seed (int): passed to numpy.random.seed

        Attributes:
            data_ (pandas.DataFrame): the data, ready for analysis
            prep (bool): flag to remember initialization settings
            yname (str): name of outcome variable
            scaler (dict): initial mean and standard error for scaling data (defaults to no scaling)
            selection_modeler (sklearn.ensemble): model class from sklearn to use for feature selection
            validation_modeler (sklearn.ensemble): model class from sklearn to use for model validation
            feature_importances_ (pandas.Series): feature importance scores from a fit model
            training_data_scaled_ (pandas.DataFrame): normalized data_
            test_data_ (pandas.DataFrame): new data for which to predict outcomes
            test_data_scaled_ (pandas.DataFrame): normalized test_data_
            predictions_ (pandas.DataFrame): predictions from model
            kfold_results_ (list): list of cross-validation results
            kfold_predictions_ (pandas.DataFrame): predctions from k-fold cross-validation
            errors_ (pandas.DataFrame): all simulated errors
            error_estimates_ (pandas.DataFrame): error estimates at specified quantiles
            simulations_ (pandas.DataFrame): simulated responses for single predictors
            coefficients_ (pandas.DataFrame): linear coefficients for simulations


        """

        self.data_ = data.copy()
        self.seed = seed
        self.scale = scale
        self.prep = prep
        self.yname = yname
        self.scaler = {'mean': 0, 'std': 1}
        if type(modeler) == tuple:
            self.selection_modeler = clone(modeler[0])
            self.validation_modeler = clone(modeler[1])
        else:
            self.selection_modeler = clone(modeler)
            self.validation_modeler = clone(modeler)
        self.feature_importances_ = pd.Series()
        self.training_data_scaled_ = pd.DataFrame()
        self.test_data_ = pd.DataFrame()
        self.test_data_scaled_ = pd.DataFrame()
        self.predictions_ = pd.DataFrame()
        self.kfold_results_ = []
        self.kfold_predictions_ = pd.DataFrame()
        self.errors_ = pd.DataFrame()
        self.error_estimates_ = pd.DataFrame()
        self.simulations_ = pd.DataFrame()
        self.coefficients_ = pd.DataFrame()

        if prep:
            move = [x not in [np.float, np.int, np.bool] for x in data.dtypes]
            ind = data.dtypes.index[move]
            if len(ind) > 0:
                self.data_ = data.copy().set_index(ind, append=True).astype('float64')

    def _save_state(self):
        """Save initializaiton parameters for future use"""

        saver = dict(
            data=self.data_.copy(), yname=self.yname, prep=self.prep, selection_modeler=clone(self.selection_modeler),
            validation_modeler=clone(self.validation_modeler))
        self.saved_state_ = dict(saver)

    def _load_state(self):
        """Load saved initialization parameters"""

        self.data_ = self.saved_state_['data']
        self.yname = self.saved_state_['yname']
        self.prep = self.saved_state_['prep']
        self.selection_modeler = self.saved_state_['selection_modeler']
        self.validation_modeler = self.saved_state_['validation_modeler']

    def _scale_data(self, input_data):
        """Subtract mean, divide by standard deviation"""

        scaled = input_data.copy()
        scaled = ((scaled - self.scaler['mean']) / self.scaler['std'])
        scaled = scaled.fillna(0)
        return scaled

    def fit(self, modeler='validation'):
        """Fit a model.

        Args:
            modeler (str): which modeler to use

        Returns:
            None: feature importances are put in a pandas.Series, ordered, and added as a class attribute.
        """

        modeler = self.validation_modeler if modeler == 'validation' else self.selection_modeler

        if self.scale:
            self.scaler = {'mean': self.data_.mean(), 'std': self.data_.std()}

        self.training_data_scaled_ = self._scale_data(self.data_)
        new_data = self.training_data_scaled_.copy()

        x_data = new_data.drop([self.yname], axis=1)
        y_data = new_data[self.yname]

        if self.seed is not None:
            np.random.seed(self.seed)
        _ = modeler.fit(X=x_data, y=y_data)

        imps = modeler.feature_importances_
        feature_importances_ = pd.Series(imps, index=x_data.columns).order(ascending=False).to_frame('value')
        feature_importances_['keep'] = True
        self.feature_importances_ = feature_importances_

    def predict(self, test_data, modeler='validation'):
        """Predict outcome given new data.

        Args:
            test_data (pandas.DataFrame): new data with same columns/shape as `data_`

        Returns:
            None: predictions DataFrame is assigned to `predictions_` attribute of class


        """

        modeler = self.validation_modeler if modeler == 'validation' else self.selection_modeler
        self.test_data_ = test_data.copy()
        new_data = self._scale_data(test_data)
        self.test_data_scaled_ = new_data.copy()

        if self.yname in new_data.columns.values.tolist():
            for_prediction = new_data.drop([self.yname], axis=1)
        else:
            for_prediction = new_data
        preds = pd.Series(index=for_prediction.index)
        preds.loc[:] = modeler.predict(for_prediction)
        if all([type(x) == pd.Series for x in self.scaler.values()]):
            preds = (preds * self.scaler['std'][self.yname]) + self.scaler['mean'][self.yname]

        actuals = self.test_data_[[self.yname]].copy()
        model_preds = pd.DataFrame(index=preds.index)
        model_preds['predicted'] = preds.astype('float64')
        model_preds['actual'] = actuals.astype('float64')

        self.predictions_ = model_preds

    def crossvalidate(self, fold_index, ahead=None, modeler='validation'):
        """Given an array of inclusion/exclusion flags, train a model, fit predictons to new data.

        Args:
            fold_index (list, np.array, pandas.Series, or pandas.Index): boolean indicators where training data
              are flagged True and testing data are flagged False
            **kwargs: passed to `self.fit`

        Returns:
            Object: a copy of the class containing the cross-validation data, model, and predictions


        """

        self._save_state()
        train = self.data_.loc[fold_index, :].copy()
        test = self.data_.loc[~fold_index, :].copy()

        if (train.std() == 0.0).any():
            zeros = (train == 0.0).values
            np.random.seed(self.seed)
            np.place(train.values, zeros, np.random.normal(0.0, 0.0000000001, zeros.sum()))

        if ahead is not None:
            test = test.iloc[[ahead], :]

        self.data_ = train
        self.fit(modeler=modeler)
        self.predict(test, modeler=modeler)
        self._load_state()
        return copy.deepcopy(self)

    def kfold(self, k=10, modeler='validation'):
        """Perform cross-validation on k randomly-partitioned sections of self.data_

        Args:
            k (int): number of folds to generate
            **kwargs: passed to crossvalidate, which passes them to fit

        Returns
            None: list of cross-validation results is stored in `self.kfold_results_` plus pandas.DataFrame of
                k-fold predictions in `kfold_predictions_`


        """

        self.kfold_results_ = None

        n = self.data_.shape[0]

        if k < 0:
            fold_indices = [np.arange(n) < i for i in np.arange(3, n + k - 1)]
            ahead = -k
        else:
            reps = (n // k) + 1
            folds = np.repeat(range(k), reps)[:n]
            np.random.seed(self.seed)
            folds = np.random.permutation(folds)
            fold_indices = [folds != fold for fold in np.unique(folds)]
            ahead = None

        fold_list = [self.crossvalidate(idx, ahead=ahead, modeler=modeler) for idx in fold_indices]

        self.kfold_results_ = fold_list
        preds = pd.concat([x.predictions_ for x in self.kfold_results_])
        self.kfold_predictions_ = preds.copy()

    def select_features(self, k=5, sensitivity=2, auto_filter=None, smooth=True, use_max=True, level_off=True):
        """Select features based on k-fold cross validation, using median absolute percent error weighted by pearson
         correlatino coefficient as a loss function

        Args:
            k (int): number of folds
            sensitivity (int): how many decimal places should be considered meaninful when comparing loss function
            auto_filter (float or int): if positive float, selection will be performed after omitting all variables
                that have importance scores less than auto_filter; if negative float, selection will be performed after
                omitting all variables where 1.0 - cumsum(variables) < auto_filter, if positive integer, selection
                will be performed on n most important variables where n == auto_filter; all other values will result
                in selection being performed on all variables
            smooth (bool): use exponentially-weighted moving average to smooth loss function results
            use_max (bool): use a moving maximum to smooth loss function results
            level_off (bool): choose optimum number of features based on the curve leveling off, rather than raw minimum

        Returns:
            None: 'keep' column of `feature_importances_` is ammended to only be True for selected variables


        """

        imps = self.feature_importances_.copy()
        to_replace = [np.inf, -np.inf, 0.]
        min_n = 1

        if auto_filter is None:
            max_n = imps.shape[0]
        elif type(auto_filter) == int:
            max_n = auto_filter if auto_filter <= imps.shape[0] else imps.shape[0]
        elif auto_filter < 0:
            vals = imps['value']
            vals = [vals.iloc[i:].sum() for i in range(vals.shape[0])]
            max_n = (np.array(vals) > -auto_filter).sum()
        elif auto_filter > 0:
            max_n = (imps['value'] > imp_boundary).sum()
        else:
            max_n = imps.shape[0]

        max_n = max_n if max_n > min_n else int(np.sqrt(imps.shape[0]))
        skip = int(np.ceil(np.sqrt(max_n - min_n)))

        saved_data = self.data_.copy()

        opts = range(min_n, max_n, skip)
        losses = pd.Series(index=opts)
        while len(opts) > 0:
            for opt in opts:
                #print opt,
                self.data_ = saved_data.copy()
                keep = [self.yname] + imps.head(opt).index.values.tolist()
                self.data_ = self.data_[keep]
                self.kfold(k=k, modeler='selection')
                preds = self.kfold_predictions_.copy()
                loss_val = ((preds['predicted'] / preds['actual']) - 1).replace(to_replace, np.nan).abs().median()
                losses[opt] = loss_val
            losses = losses.sort_index()
            anchors = losses.iloc[1:][losses.iloc[1:].round(sensitivity).diff().shift(-1) > 0.0].index.values
            pool = losses.index.values.tolist()
            opts = []
            for a in anchors:
                i = pool.index(a)
                start_i = i - 1
                end_i = i + 1
                prange = range(pool[start_i] + 1, pool[i]) + range(pool[i] + 1, pool[end_i])
                n = int(np.ceil(np.sqrt(len(prange))))
                if len(prange) > 1:
                    opts += np.random.choice(prange, n, replace=False).tolist()
            opts = [x for x in opts if x not in losses.index.values]
            opts = sorted(opts)

        losses = losses.reindex_like(pd.Series(index=range(min_n, max_n + 1))).interpolate()
        if smooth:
            losses = pd.ewma(losses, 3)
        if use_max:
            losses = pd.rolling_max(losses, 3, center=True)

        if level_off:
            keep_n = (losses.round(sensitivity).diff() >= 0.0).argmax()
        else:
            keep_n = losses.round(sensitivity).argmin()

        imps.iloc[keep_n:, :]['keep'] = False

        self.data_ = saved_data.copy()
        self.feature_importances_ = imps.copy()

    def filter_data(self, new_data=None):
        """Given `feature_importances_`, keep only outcome and selected features in `data_`"""

        imps = self.feature_importances_.copy()
        keep = [self.yname] + imps[imps['keep']].index.values.tolist()

        if new_data is None:
            self.data_ = self.data_[keep].copy()
        else:
            return new_data[keep].copy()

    def empirical_error(self, predictions=None, cv_preds=None, cv_actuals=None, n=1000, q=None):
        """Calculate prediction errors based on cross-validation errors.

        Args:
            predictions (pd.Series, optional): predictions from model; taken from `predictions_` if None
            cv_preds (pandas.Series, optional): predictions from cross validation; taken from kfold_predictions_ if None
            cv_actuals (pandas.Series, optional): values from cross validation; taken from kfold_predictions_ if None
            n (int): number of error metrics to randomly sample
            q (float, optional): probability for which to calculate credibile interval

        Returns:

            None: if q is None, assigns data frame with estimates and all errors to `errors_`;
                if q is not None, assigns estimates and credible interval boundaries to `error_estimates_`


        """

        return_output = predictions is not None
        cv_preds = self.kfold_predictions_['predicted'].copy() if cv_preds is None else cv_preds
        cv_actuals = self.kfold_predictions_['actual'].copy() if cv_actuals is None else cv_actuals

        if (cv_preds == 0.0).all():
            cv_preds.loc[:] = np.random.normal(0.0, 0.0000000001, cv_preds.shape[0])
        if (cv_actuals == 0.0).all():
            cv_actuals.loc[:] = np.random.normal(0.0, 0.0000000001, cv_actuals.shape[0])

        predictions = self.predictions_['predicted'].copy() if predictions is None else predictions

        # calculate ratio of predicted and actual
        diffs = cv_preds.astype(float) / cv_actuals.astype(float)
        diffs = pd.Series(diffs).replace([np.inf, -np.inf, 0.0], np.nan).dropna().values

        # scale predictions by randomly-selected error ratios
        error_list = [p * np.array(sorted(np.random.choice(diffs, n))) for p in predictions]
        errors = np.column_stack(error_list)
        idx = predictions.index if hasattr(predictions, 'index') else range(errors.shape[1])
        errors = pd.DataFrame(errors, columns=idx)
        if not return_output:
            self.errors_ = errors

        if q is not None:
            for prob in q:
                output = pd.DataFrame({'estimate': predictions})
                p = (1.0 - prob) / 2
                output['lwr' + '{:0.0f}'.format(prob*100).replace('0.', '')] = errors.quantile(0.0 + p)
                output['upr' + '{:0.0f}'.format(prob*100).replace('0.', '')] = errors.quantile(1.0 - p)
            if not return_output:
                self.error_estimates_ = output.copy()
        if return_output:
            return (output, errors) if q is not None else errors

    def simulate(self, predictors=None, n_compare=50, n_sims=500, method=None):
        """Simulate response of outcome variable to changes in each predictor, holding everything else constant.

        Args:
            predictors (list, optional): list of predictors for which to calculate coefficients, all predictors in
                `data_` used if none fed to method
            n_sims (int): number of values to simulate from original data set
            n_compare (int): number of error-weighted comparisons to make for each simulation
            method (list): method(s) to use for calculating correlation; can be 'pearson', 'kendall', and/or 'spearman';
                if None (default), only 'pearson' is used

        Returns:
            simulations_ (pandas.DataFrame):
            coefficients_ (pandas.DataFrame):


        """

        predictors = self.data_.drop([self.yname], axis=1).columns.values.tolist() if predictors is None else predictors
        sim_data = self.training_data_scaled_.copy()
        sim_inds = np.random.choice(sim_data.shape[0], n_sims)
        sim_data = sim_data.iloc[sim_inds, :]
        sim_vals = sim_data.iloc[np.repeat(range(n_sims), n_compare), :].copy()

        preds = pd.Series(index=range(sim_data.shape[0]))
        preds.loc[:] = self.validation_modeler.predict(sim_data.drop([self.yname], axis=1))
        cv_preds = self.kfold_predictions_.copy()
        preds = self.empirical_error(preds, cv_preds['predicted'], cv_preds['actual'], n=n_compare).unstack()
        sim_vals[self.yname] = preds.values

        methods = ['pearson'] if method is None else method
        output = pd.DataFrame(index=predictors)
        for m in methods:
            output[m] = pd.concat([sim_vals[[p, self.yname]].corr(m).iloc[[0], [-1]] for p in predictors])
        output['importance'] = self.feature_importances_['value']
        output = output.sort('importance', ascending=False)

        self.simulations_ = sim_vals
        self.coefficients_ = output

    def run(
            self, ks=(5, 10), sensitivity=2, auto_filter=None, smooth=True, use_max=True, level_off=True,
            predictors=None, n_sims=500, n_compare=50, verbose=False):
        """

        Args:
            ks (tuple): length 2, first element fet to `select_features`, second to `kfold`
            minimum (int): passed to `select_features`
            scaler (float): passed to `select_features`

        Returns:
            None


        """

        if verbose:
            t_base = time.time()
            print 'running:',

        self.fit()

        if verbose:
            print 'selecting...',
            t0 = time.time()

        self.select_features(
            k=ks[0], sensitivity=sensitivity, auto_filter=auto_filter, smooth=smooth, use_max=use_max,
            level_off=level_off)
        self.filter_data()

        if verbose:
            print self.feature_importances_['keep'].sum(),
            print round(time.time() - t0, 1),
            print 'validating...',
            t0 = time.time()

        self.kfold(k=ks[1])
        self.fit()

        if verbose:
            print round(time.time() - t0, 1),
            print 'simulating...',
            t0 = time.time()

        self.simulate(predictors=predictors, n_sims=n_sims, n_compare=n_compare)

        if verbose:
            print round(time.time() - t0, 1),
            print 'total time:', round(time.time() - t_base, 1),
            print 'finishing...'


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
        """
        Args:
            method: 'division' or 'lowess'
            cutoff: threshold for binarizing reults of transforming self.value_var by method
            **kwargs: passed to smoother.lowess


        """
        
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


class DimensionalityReduction(object):
    """Class for doing demensionality reduction through principal components analysis, with options for kernel
    transformation and affinity propagation.


    """

    def __init__(self, data):
        """Initialization function.

        Args:
            data (pandas.DataFrame): data frame of matrix to be decomposed (all non-number columns must be in index)

        Attributes:
            data_ (pandas.DataFrame): raw data
            corrmatrix_ (pandas.DataFrame): correlation matrix of data_
            variance_explained_ (pandas.DataFrame): variance explained for filtered components
            all_variance_explained_ (pandas.DataFrame): variance explained for all components
            loadings_ (pandas.DataFrame): filtered component loadings
            all_loadings_ (pandas.DataFrame): all component loadings
            projection_ (pandas.DataFrame): principal components transformation of data_
            pca_object_ (sklearn.decomposition.KernelPCA): KernelPCA class
            ap_object_ (sklearn.cluster.AffinityPropagation): Affinity Propagation class
            clusters_ = (pandas.Series): cluster assignments from affinity propagation


        """
        self.data_ = data.copy()
        self.corrmatrix_ = pd.DataFrame()
        self.variance_explained_ = pd.DataFrame()
        self.all_variance_explained_ = pd.DataFrame()
        self.loadings_ = pd.DataFrame()
        self.all_loadings_ = pd.DataFrame()
        self.projection_ = pd.DataFrame()
        self.pca_object_ = None
        self.ap_object_ = None
        self.clusters_ = pd.Series()

    def prep(self, log=False, standardize=True, method='pearson', nas=None):
        """Transform data_ into correlation matrix.

        Args:
            log (bool): whether to take natural logarithm of data before standardizing (mins are subtracted)
            standardize (bool): whether to z-score data before correlating
            method (str): passed to pandas.DataFrame.corr
            nas (str or int or float): how to handle null values

        Returns:
            corrmatrix_ (pandas.DataFrame): correlation matrix


        """
        df = self.data_.copy()
        if log:
            df = np.log(df - df.min() + 1)

        if type(standardize) is not bool:
            zscore = lambda x: (x - x.mean()) / x.std()
            df = df.groupby(level=standardize).transform(zscore)
        elif standardize:
            df = (df - df.mean()) / df.std()

        if method == 'jaccard':
            if not np.all(df.dtypes == bool):
                df = df.T.notnull()
            jaccard = pairwise_distances(df.values, metric='jaccard')
            df = pd.DataFrame(jaccard, columns=df.columns, index=df.columns)
        elif method is not None:
            df = df.corr(method)

        if nas == 'drop_1':
            df = df.dropna(axis=1)
        elif nas == 'drop_0':
            df = df.dropna(axis=0)
        elif nas == 'drop':
            df = df.dropna()
        elif nas == 'all':
            ind = df.isnull().all().values
            df = df.loc[~ind, :].loc[:, ~ind]
        elif nas is not None:
            df = df.fillna(nas)

        self.corrmatrix_ = df.copy()

    def decompose(self, kernel='linear', method='pearson', threshold=2, **kwargs):
        """

        Args:
            kernel (str): passed to KernelPCA
            method (str): passed to pandas.DataFrame.corr
            threshold (int): cutoff for filtering components
            kwargs: passed to KernelPCA

        Returns:
            variance_explained_ (pandas.DataFrame): variance explained for filtered components
            all_variance_explained_ (pandas.DataFrame): variance explained for all components
            loadings_ (pandas.DataFrame): filtered component loadings
            all_loadings_ (pandas.DataFrame): all component loadings
            projection_ (pandas.DataFrame): principal components transformation of data_
            pca_object_ (sklearn.decomposition.KernelPCA): KernelPCA class


        """
        sp = KernelPCA(n_components=min(*self.corrmatrix_.shape), kernel=kernel, **kwargs)

        proj = sp.fit_transform(self.corrmatrix_)
        proj = pd.DataFrame(proj, index=self.corrmatrix_.index)
        proj.columns = [x+1 for x in proj.columns]

        var_expl = sp.lambdas_
        var_expl[var_expl < 0.0] = 0.0
        var_expl = var_expl / var_expl.sum()

        loadings = pd.concat([self.corrmatrix_, proj], axis=1).corr(method)
        loadings = loadings.loc[:, proj.columns].loc[self.corrmatrix_.columns, :]
        self.all_loadings_ = loadings.copy()
        keep = (var_expl > (float(threshold) / self.corrmatrix_.shape[1])).sum()
        loadings = loadings.iloc[:, :keep]

        variance_explained = pd.DataFrame(index=loadings.columns)
        self.all_variance_explained_ = variance_explained.copy()
        variance_explained['total'] = var_expl[:keep]
        variance_explained['cumulative'] = var_expl.cumsum()[:keep]
        variance_explained['max_abs_corr'] = loadings.abs().max()

        self.variance_explained_ = variance_explained.copy()
        self.loadings_ = loadings.copy()
        self.projection_ = proj.copy()
        self.pca_object_ = sp

    def _evaluate_propagation(self, loads, p, d, weights=True, **kwargs):
        """Helper function to aid with grid search of AffinityPropagation parameters.

        Args:
            loads (pandas.DataFrame): loadings from KernelPCA
            p (int): passed to AffinityPropagation
            d (int): passed to AffinityPropagation
            weights (bool): whether to weight summary measures by the number of records in a grouping
            **kwargs: passed to AffinityPropagation

        Returns:
            rang (float): (possibly weighted) average range of loadings
            percents (float): (possibly weighted) average of percentage of loadings that have the same sign as
            the majority of records in the grouping


        """

        cols = loads.columns
        km = AffinityPropagation(preference=p, damping=d, **kwargs)
        x = km.fit(loads.values)
        labels = pd.Series(km.labels_).value_counts(normalize=True).sort_index()
        maxs = self.loadings_[cols].groupby(km.labels_).max()
        mins = self.loadings_[cols].groupby(km.labels_).min()
        rang = maxs - mins
        rang = (rang.T * labels).T.sum() / labels.sum()

        percent_majority = lambda x: np.sign(x).apply(lambda y: y.value_counts(normalize=True).max())
        percents = self.loadings_[cols].groupby(km.labels_).apply(percent_majority)

        if weights:
            use_weights = self.variance_explained_['total'][cols].values
            rang = (rang * np.array(use_weights)).sum() / np.array(use_weights).sum()
            percents = ((percents * np.array(use_weights)).sum(axis=1) / np.array(use_weights).sum()).mean()
        else:
            rang = rang.mean()
            percents = percents.mean().mean()

        return rang, percents

    @staticmethod
    def check_continuity(x, sensitivity=2):
        """Returns midpoint of largest grouping of lowest values (rounded to specified decimal points).

        Args:
            x (pandas.Series): values are diagnostic outputs, index is parameter values from grid search
            sensitivity (int): how many decimal places to consider when looking for best parameters

        Returns:
            x_keep (float): best value of x.index


        """

        x = pd.Series(x, index=x.index).round(sensitivity)
        x_min = x.min()
        j = 0
        outs = []
        for i in range(1, len(x)):
            if x.iloc[i] != x.iloc[i-1]:
                j += 1
            outs.append(j)
        x_groups = pd.Series([0] + outs, index=x.index)
        group_ind = x_groups[x == x_min].value_counts().argmax()
        in_group = (x_groups == group_ind).values
        x_keep = np.sum(x.index.values[in_group][[0, -1]]) / 2
        return x_keep

    def affinity_propagation(self, n_inputs=100, weights=True, sensitivity=2, subset=None, **kwargs):
        """Use Affinity Propagation to cluster records based on PCA loadings.

        Args:
            n_inputs: number of values to test for `preference` parameter in AffinityPropagation
            weights (bool): whether to weight summary measures by the number of records in a grouping
            sensitivity (int): how many decimal places to consider when looking for best parameters
            **kwargs: passed to AffinityPropagation

        Returns:
            ap_object_ (sklearn.cluster.AffinityPropagation): Affinity Propagation class
            clusters_ = (pandas.Series): cluster assignments from affinity propagation

        Notes:
            good defaults: kwargs = dict(convergence_iter=30, max_iter=400)


        """

        if type(subset) == int:
            loads = self.loadings_.loc[:, :subset].fillna(0.0)
        elif hasattr(subset, '__iter__'):
            loads = self.loadings_.loc[:, subset].fillna(0.0)
        else:
            loads = self.loadings_.fillna(0.0)

        dists = pairwise_distances(loads.values).flatten() ** 2
        dists = dists[dists > 0.0]
        inputs = np.linspace(-np.median(dists) / 2, -dists.min(), n_inputs + 1)[:-1]
        ds = np.arange(0.5, 1.0, .01)

        out = [self._evaluate_propagation(loads, p=inputs[0], d=d, weights=weights, **kwargs) for d in ds]
        mean_range, mean_percent = zip(*out)
        mean_range = pd.Series(mean_range, index=ds)
        mean_percent = pd.Series(mean_percent, index=ds)
        out_range = self.check_continuity(mean_range, sensitivity)
        out_percent = self.check_continuity(mean_percent, sensitivity)
        damp = np.mean([out_range, out_percent])

        out = [self._evaluate_propagation(loads, p=i, d=damp, weights=weights, **kwargs) for i in inputs]
        mean_range, mean_percent = zip(*out)
        mean_range = pd.Series(mean_range, index=inputs)
        mean_percent = pd.Series(mean_percent, index=inputs)
        out_range = self.check_continuity(mean_range, sensitivity)
        out_percent = self.check_continuity(mean_percent, sensitivity)
        pref = np.mean([out_range, out_percent])

        km = AffinityPropagation(preference=pref, damping=damp, **kwargs)
        x = km.fit(loads.values)

        self.ap_object_ = km
        self.clusters_ = pd.Series(km.labels_, index=self.loadings_.index)
