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

    def __init__(self, data, yname, prep=False):
        """Load data and set up basic model parameters

        Args:
            data (pandas.DataFrame): the data to train the model
            yname (str): the name of the column representing the outcome variable
            prep (bool): whether to ensure the dataset has only float values in it

        Attributes:
            data_ (pandas.DataFrame): the data, ready for analysis
            prep (bool): flag to remember initialization settings
            yname (str): name of outcome variable
            scaler (dict): initial mean and standard error for scaling data (defaults to no scaling)
            modeler (sklearn.ensemble): model class from sklearn
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
        self.prep = prep
        self.yname = yname
        self.scaler = {'mean': 0, 'std': 1}
        self.modeler = None
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

        saver = dict(data=self.data_.copy(), yname=self.yname, prep=self.prep, modeler=clone(self.modeler))
        self.saved_state_ = dict(saver)

    def _load_state(self):
        """Load saved initialization parameters"""

        self.data_ = self.saved_state_['data']
        self.yname = self.saved_state_['yname']
        self.prep = self.saved_state_['prep']
        self.modeler = self.saved_state_['modeler']

    def _scale_data(self, input_data):
        """Subtract mean, divide by standard deviation"""

        scaled = input_data.copy()
        scaled = ((scaled - self.scaler['mean']) / self.scaler['std'])
        scaled = scaled.fillna(0)
        return scaled

    def load_modeler(self, obj):
        """Load instantiated sklean.ensemble class"""

        self.modeler = clone(obj)

    def fit(self, scale=True, seed=None):
        """Fit a model.

        Args:
            scale (bool): whether to normalize the data before fitting
            seed (int, optional): number to seed the pseudo-random number generator

        Returns:
            None: feature importances are put in a pandas.Series, ordered, and added as a class attribute.
        """

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
        feature_importances_ = pd.Series(imps, index=x_data.columns).order(ascending=False).to_frame('value')
        feature_importances_['keep'] = True
        self.feature_importances_ = feature_importances_

    def predict(self, test_data):
        """Predict outcome given new data.

        Args:
            test_data (pandas.DataFrame): new data with same columns/shape as `data_`

        Returns:
            None: predictions DataFrame is assigned to `predictions_` attribute of class


        """
        self.test_data_ = test_data.copy()
        new_data = self._scale_data(test_data)
        self.test_data_scaled_ = new_data.copy()

        if self.yname in new_data.columns.values.tolist():
            for_prediction = new_data.drop([self.yname], axis=1)
        else:
            for_prediction = new_data
        preds = pd.Series(index=for_prediction.index)
        preds.loc[:] = self.modeler.predict(for_prediction)
        if all([type(x) == pd.Series for x in self.scaler.values()]):
            preds = (preds * self.scaler['std'][self.yname]) + self.scaler['mean'][self.yname]

        actuals = self.test_data_[[self.yname]].copy()
        model_preds = pd.DataFrame(index=preds.index)
        model_preds['predicted'] = preds.astype('float64')
        model_preds['actual'] = actuals.astype('float64')

        self.predictions_ = model_preds

    def crossvalidate(self, fold_index, **kwargs):
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
        self.data_ = train
        self.fit(**kwargs)
        self.predict(test)
        self._load_state()
        return copy.deepcopy(self)

    def kfold(self, k=10, **kwargs):
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
        reps = (n//k)+1
        folds = np.repeat(range(k), reps)[:n]
        folds = np.random.permutation(folds)
        fold_list = [self.crossvalidate(folds != fold, **kwargs) for fold in np.unique(folds)]

        self.kfold_results_ = fold_list
        preds = pd.concat([x.predictions_ for x in self.kfold_results_])
        self.kfold_predictions_ = preds.copy()

    def select_features(self, k=2, boundary=100, minimum=2, sensitivity=2, **kwargs):
        """Select features based on k-fold cross validation, using median absolute percent error as a loss function

        Args:
            k (int): number of folds
            boundary (int): if number of features chosen after first pass is greater than boundary, second pass
                will explore every number between the first-pass results and first-pass results / 2; otherwise, it
                will explore every number between first-pass results and 1
            minimum (int): minimum number of features to include
            sensitivity (int): how many decimal places to consider when comparing loss function results
            **kwargs: passed to `kfold`, which is passed to `cross_validate`, which is passed to `fit`

        Returns:
            None: 'keep' column of `feature_importances_` is ammended to only be True for selected variables


        """

        to_replace = [np.inf, -np.inf, 0.]
        imps = self.feature_importances_.copy()
        n = imps.shape[0]
        ns = []
        self._save_state()
        saved_data = self.saved_state_['data'].copy()
        while n > minimum:
            keep = [self.yname] + imps.head(n).index.values.tolist()
            self.data_ = self.data_[keep]
            self.kfold(k=k, **kwargs)
            preds = self.kfold_predictions_.copy()
            loss_mean = ((preds['predicted'] / preds['actual']) - 1).replace(to_replace, np.nan).abs().median()
            ns.append((n, loss_mean))
            #print n, round(loss_mean, 4)
            n = int(n/2)

        self.saved_state_['data'] = saved_data.copy()
        self._load_state()

        n_ind = pd.Series([x[1] for x in ns]).pct_change().argmin() - 1
        max_n = pd.Series([x[0] for x in ns])[n_ind] + 1
        min_n = minimum if max_n < boundary else max_n/2.0

        ns = []
        for n in range(min_n, max_n)[::-1]:
            keep = [self.yname] + imps.head(n).index.values.tolist()
            self.data_ = self.data_[keep]
            self.kfold(k=k, **kwargs)
            preds = self.kfold_predictions_.copy()
            loss_mean = ((preds['predicted'] / preds['actual']) - 1).replace(to_replace, np.nan).abs().median()
            ns.append((n, loss_mean))
            #print n, round(loss_mean, 4)

        self.saved_state_['data'] = saved_data.copy()
        self._load_state()

        keep_n = pd.Series([x[1] for x in ns]).round(sensitivity).argmin()
        keep_n = pd.Series([x[0] for x in ns])[keep_n]

        imps.ix[keep_n:, 'keep'] = False
        self.feature_importances_ = imps.copy()

    def filter_data(self, new_data=None):
        """Given `feature_importances_`, keep only outcome and selected features in `data_`"""

        imps = self.feature_importances_.copy()
        keep = [self.yname] + imps[imps['keep']].index.values.tolist()

        if new_data is None:
            self.data_ = self.data_[keep].copy()
        else:
            return new_data[keep].copy()

    def simulate(self, predictors=None, n_inc=100, n_compare=100):
        """Simulate response of outcome variable to changes in each predictor, holding everything else constant.

        Args:
            predictors (list, optional): list of predictors for which to calculate coefficients, all predictors in
                `data_` used if none fed to method
            n_inc (int): number of evenly-spaced increments in the outcome variable to use for the simulation
            n_compare (int): number of comparisons to make for each increment

        Returns:
            simulations_ (pandas.DataFrame):
            coefficients_ (pandas.DataFrame):


        """

        predictors = self.data_.drop([self.yname], axis=1).columns.values.tolist() if predictors is None else predictors
        sim_data = self.training_data_scaled_.copy()

        output_dfs = []
        boots = []
        for p in predictors:
            sims = np.linspace(sim_data[p].min(), sim_data[p].max(), n_inc)
            others = [pred for pred in predictors if pred != p]
            weights = [make_weights(sim_data[p], (sims < s).mean()) for s in sims]
            other_vals = [[np.random.choice(sim_data[o], size=n_compare, p=w) for o in others] for w in weights]
            other_vals = np.concatenate([np.column_stack(vals) for vals in other_vals], axis=0)

            idx = pd.MultiIndex.from_tuples(others, names=self.data_.columns.names)
            sim_vals = pd.DataFrame(other_vals, columns=idx)
            sim_vals[p] = sims.tolist() * n_compare
            sim_vals = sim_vals[predictors]
            sim_vals = sim_vals.sort([p])

            predictor = sim_vals[p].copy()
            preds = pd.Series(index=sim_vals.index)
            preds.loc[:] = self.modeler.predict(sim_vals)

            #plt.plot(predictor, preds, '.')
            #plt.show()

            boot_output = pd.Series()
            boot = []
            for i in range(n_compare):
                resample = np.random.choice(predictor.shape[0], predictor.shape[0])
                x = predictor.to_frame('p').iloc[resample, :]
                y = preds.iloc[resample]
                slope = LinearRegression().fit(X=x, y=y).coef_[:][0]
                boot.append(slope)
            boot_output['smean'] = pd.Series(boot).mean()
            boot_output['sstd'] = pd.Series(boot).std()
            check = preds.groupby(predictor).mean()
            boot_output['soffset'] = LinearRegression().fit(X=check.to_frame('p'), y=check.index.values).coef_[:][0]

            if all([type(x) == pd.Series for x in self.scaler.values()]):
                preds = (preds * self.scaler['std'][self.yname]) + self.scaler['mean'][self.yname]
                predictor = (predictor * self.scaler['std'][p]) + self.scaler['mean'][p]
            output = pd.DataFrame(index=sim_vals.index)
            output['outcome'] = preds
            output['value'] = predictor
            index_names = sim_vals.columns.names + ['increment', 'simulation']
            increment = np.tile(range(n_inc), n_compare)
            simulation = np.repeat(range(n_compare), n_inc)
            output_index = [p + (increment[j], ) + (simulation[j], ) for j in range(len(increment))]
            output_index = pd.MultiIndex.from_tuples(output_index, names=index_names)
            output.index = output_index

            boot = []
            for i in range(n_compare):
                resample = np.random.choice(predictor.shape[0], predictor.shape[0])
                x = predictor.to_frame('p').iloc[resample, :]
                y = preds.iloc[resample]
                slope = LinearRegression().fit(X=x, y=y).coef_[:][0]
                boot.append(slope)
            boot_output['rmean'] = pd.Series(boot).mean()
            boot_output['rstd'] = pd.Series(boot).std()
            check = preds.groupby(predictor).mean()
            boot_output['roffset'] = LinearRegression().fit(X=check.to_frame('p'), y=check.index.values).coef_[:][0]

            output_dfs.append(output.copy())
            boots.append(boot_output.copy())

        output_dfs = pd.concat(output_dfs, axis=0)
        boots = pd.concat(boots, axis=1, keys=predictors).T
        boots.index = pd.MultiIndex.from_tuples(boots.index.values, names=self.data_.columns.names)
        boots['importance'] = self.feature_importances_[self.feature_importances_['keep']]['value']

        self.simulations_ = output_dfs
        self.coefficients_ = boots

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


def corr_data(data, i, value_var, time_var, index_vars, min_time=None, t=None, weight=False, fill=True, return_df=True):
    t0 = time.time()
    print 'shaping data sets...'
    
    min_t = data[time_var].max() if min_time is None else min_time
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
    corrs = corrs.iloc[:col_ind, (col_ind+1):]
    corrs = corrs.fillna(0)
    if weight:
        jaccard = jaccard.iloc[:col_ind, (col_ind+1):]
        jaccard = 1 - jaccard
        corrs *= jaccard
    column_names = [str(x)+'_compare' for x in corrs.columns.names]
    corrs.columns.names = column_names
    
    if t is not None:        
        corrs = corrs.stack(column_names)
        corrs = corrs[corrs >= t].sort_index()
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
        if d[i-1] != d[i]:
            j = i
        out.append(j)
    
    s = pd.Series(out).value_counts()
    if s.iloc[0] == 1:
        print 'no good choice'
    else:
        return arr.index[s.index[0]]

def percent_majority(x):
    signed = np.sign(x)
    top = np.sign(x).astype(str).describe().T['top'].astype(float)
    percent = (signed != top).sum()
    return percent

def affinity_propagation(df, weights=None, rounder=3, output='object', 
                         verbose=False):
    if type(df) == pd.DataFrame:
        df1 = df.values
    else:
        df1 = df
    dists = pairwise_distances(df1).flatten()**2
    dists = dists[dists > 0.0]
    inputs = np.linspace(-np.median(dists)/2, -dists.min(), 101)[:-1]
    ds = np.arange(0.5, 1.0-0.01, .01)

    out = []
    if verbose:
        print 'exploring damping options...'
    for d in ds:
        if verbose:
            print d, 
        km = AffinityPropagation(preference=inputs[0], damping=d, convergence_iter=30, max_iter=400)
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

    out = pd.DataFrame({'mean_range': out, 'total_count': counts, 'n_groups': n}, index=inputs)
    
    min_ind = get_continuity_start(out['n_groups'].pct_change(),rounder)
    max_ind = (out.loc[min_ind:, 'n_groups'].pct_change().dropna().abs().round(rounder) != 0.0).argmax()
    pref = out.loc[min_ind:max_ind, :].iloc[:-1, :]
    if len(pref)==0:
        print ''
        print 'mean_corr and n_mult criteria exclude all possibilities'
        return out
    else: 
        pref = pref[pref['total_count'] == pref['total_count'].min()].index[0]
    
    if output == 'object':
        km = AffinityPropagation(preference=pref, damping=damp, convergence_iter=30, max_iter=400)
        x = km.fit(df1) 
        return km
    else:
        return damps, out, damp, pref


def principal_components(
        data, idx, val, kernel='linear', cluster=True, components=None, sensitivity=3, verbose=False, output_dict=None):
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
        ap = affinity_propagation(
            df=loadings[load_cols], weights=variance_explained['total'][load_cols].values, rounder=sensitivity,
            verbose=verbose)
        output['cluster'] = ap.labels_

        if output_dict is not None:
            level_values = output.index.get_level_values(output_dict).values
            cat_dict = pd.Series(level_values).groupby(ap.labels_).unique()
            cat_dict = cat_dict.to_dict()
            cat_dict = {k:v.tolist() for k,v in cat_dict.items()}
        
    class Final(object):
        correlation = df.copy()
        diagnostic = variance_explained
        pca_object = sp
        ap_object = ap
        loading = output
        cluster_dictionary = cat_dict
    
    final_output = Final()
    
    return final_output


def plot_components(loadings, labels=None, components=None):
    loads = loadings.copy()
    if labels is not None:
        loads['cluster'] = labels

    x_lab = components[0] if components is not None else 1
    y_lab = components[1] if components is not None else 2

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