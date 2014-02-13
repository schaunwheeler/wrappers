# -*- coding: utf-8 -*-

import os
import inspect
import sys
import warnings
import pandas as pd
import numpy as np
import itertools


def set_module_path(module_name, guide='user', choose_max=True):
    '''Allows a custom script to be imported as a module from outside
    the current working directory. This will only modify the path for the
    current session.

    Parameters
    ----------

    module_name : string, the name of the module to be imported
    guide : if 'user', will search all folders starting from the user directory,
        if 'wd', will search only within the working directory; otherwise, guide
        must be the full path to the module folder
    choose_max : if guide is 'user' or 'wd', and the search returns more than
        one folder, if choose_max == True the longest of the paths will be used.


    '''

    if guide == 'user':
        cmd_folder = os.path.expanduser('~')
    elif guide == 'wd':
        cmd_folder = os.path.realpath(os.path.abspath(os.path.split(
            inspect.getfile(inspect.currentframe()))[0]))
    else:
        module_folder = guide
        print 'assuming `guide` gives direct path to module'

    if guide in ['user', 'wd']:
        module_folders = []
        for root, dirs, files in os.walk(cmd_folder, topdown=False):
            for name in dirs:
                if name.endswith(module_name):
                    module_folders.append(root)

        if len(module_folders) > 1:
            print 'Multiple roots were found:'
            for root in module_folders:
                print '  ', root

        if (len(module_folders) == 1) | (choose_max == True):
            module_folder = max(module_folders)
        else:
            module_folder = None

    if module_folder == None:
        print 'no module inserted into path'
    elif module_folder not in sys.path:
        sys.path.insert(0, module_folder)
        print module_folder, 'inserted into path'
    elif module_folder in sys.path:
        print module_folder, 'already in path'
    else:
        print 'could not determine what to do'

def trim_matrix(df):
    '''
    Takes a correlation data frame and removes all row/column
    combinations where the correlation is np.nan. For use when
    missing correlations mess up a cluster analysis.
    '''

    null_mask = df.apply(lambda x: x.isnull().mean()) == 0.0
    df = df.ix[null_mask, null_mask]
    return df


def write_dataframe(df, conn, table_name, step=None, table_key=None,
    table_index=None, skip_errors=True, action=['prepare', 'append'],
    deconflict='dataframe', flavor='sqlite', verbose=True):
    ''' Write data frame in chunks to avoid sql connectivity issues.

    df : the data frame to be written the database
    conn : the database connection to which to write the dataframe
    step : the number of rows from df to write at a time.
    table_name : the table within the database to which to write the df
    table_key : a list of columns that form unique identifiers for each row
    table_index : a dictionary of lists, where the keys identify the names of
        indices to be created, and the lists within the dictionary values
        contain the names of the columns that will make up each dictionary
    skip_errors : boolean, indicates whether to continue the upload even if
        errors occur
    action : list, whether to prepare a table for upload, upload a data frame,
        or both
    deconflict: if 'database', remove rows from database that have the same
        key as rows in the dataframe; if 'dataframe', remove rows in the data
        frame that have the same key as rows in the database; if 'none', do
        nothing
    flavor : passed to pandas.io.sql, indicating the flavor of SQL to expect
    verbose : boolean, indicating whether to display informative progress
        messages

    '''
    df = df.convert_objects(convert_numeric=True)

    if ('append' in action) and ('prepare' not in action) and deconflict != 'none':
        if verbose:
            print 'Reading "%s" table to deconflict data frame.' % table_name

        if len(table_key) == 1:
            concat_statement = table_key[0]
        elif flavor == 'mysql':
            concat_statement = 'CONCAT(%s)' % ', '.join(table_key)
        elif flavor in ['postgresql', 'sqllite', 'oracle']:
            concat_statement =  '||'.join(table_key)
        else:
            concat_statement = '+'.join(table_key)

        previous_index = pd.io.sql.read_frame(
            'SELECT %s from %s' % (concat_statement, table_name), conn).squeeze()
        new_index = df[table_key].fillna('').applymap(str).apply(
            lambda x: ''.join(x), axis=1)

        if deconflict == 'dataframe':
            df = df[~new_index.isin(previous_index)]
        elif deconflict == 'database':
            to_remove = previous_index[previous_index.isin(new_index)].values
            remove_statement = 'DELETE FROM %s WHERE %s IN (%s)' % (table_name,
                concat_statement, ', '.join(to_remove))

            cursor = conn.cursor()

            try:
                cursor.execute(remove_statement)
            except conn.Error:
                cursor.close()
                cursor = conn.cursor()
                cursor.execute(remove_statement)
            cursor.close()

    if 'prepare' in action:
        if pd.io.sql.table_exists(table_name, conn, "mysql") & verbose:
            print 'Table named "%s" already exists.' % table_name
            print 'Existing table will be replaced.\n'

        if verbose:
            print 'Preparing query...'

        def check_digits(series):
            if series.dtype.name in ['float64']:
                diff = series - series.round(0)
                if np.nanmax(diff) == 0.0:
                    series = series.fillna(0.0).astype('int64')
            return series

        def check_lengths(x):
            if x.dtype == np.dtype('O'):
                out = x.str.len().max()
            else:
                out = x.apply(str).str.len().max()
            return out

        df = df.apply(check_digits)
        bool_cols = df.dtypes == 'bool'
        df.ix[:,bool_cols] = df.ix[:,bool_cols].apply(
            lambda x: x.astype('int64'))

        datatype_dict = {np.dtype('float64'):'FLOAT(%d)',
                         np.dtype('int64'):'INT(%d)',
                         np.dtype('O'):'VARCHAR(%d)'}
        sql_dtypes = df.dtypes.map(datatype_dict)
        col_maxlengths = df.apply(check_lengths)

        sql_dtypes = pd.Series(
            [sql_dtypes[i] % col_maxlengths[i] for i in range(sql_dtypes.shape[0])],
            index = sql_dtypes.index)

        sql_dtypes = pd.DataFrame({
            'column_name': sql_dtypes.index.values,
            'data_types': sql_dtypes,
            'not_null': np.where(sql_dtypes.index.isin(table_key), 'NOT NULL', '')
        })

        query = [
            'DROP TABLE IF EXISTS %s;' % table_name,
            'SHOW WARNINGS;',
             #'SET GLOBAL max_allowed_packet=1073741824;'
            'CREATE  TABLE IF NOT EXISTS %s (' % table_name]

        for i in range(sql_dtypes.shape[0]):
            query.append(
                '%s %s %s REFERENCES %s (%s),' % (
                    sql_dtypes['column_name'][i], sql_dtypes['data_types'][i],
                    sql_dtypes['not_null'][i], table_name,
                    sql_dtypes['column_name'][i]))

        query.append('created TIMESTAMP DEFAULT CURRENT_TIMESTAMP,')
        query.append('updated TIMESTAMP ON UPDATE CURRENT_TIMESTAMP,')

        if table_key is None:
            table_key = df.columns[0]

        query.append('PRIMARY KEY (%s))' % ', '.join(table_key))
        query.append('CHARSET utf8 COLLATE utf8_general_ci;')

        if table_index is not None:
            for idx, cols in table_index.iteritems():
                query.append('CREATE INDEX %s on %s (%s);' % (idx, table_name,
                             ', '.join(cols)))

        query.append('SHOW WARNINGS;')

        query = ' '.join(query)

        cursor = conn.cursor()

        try:
            cursor.execute(query)
        except conn.Error:
            cursor.close()
            cursor = conn.cursor()
            cursor.execute(query)
        cursor.close()

    if 'append' in action:

        df = df.where((pd.notnull(df)), None)

        if step is None:
            step = df.shape[0] - 1

        if verbose:
            print 'Executing query...'

        with warnings.catch_warnings():
            warnings.simplefilter('ignore', conn.Warning)
            if verbose:
                print 'Records (out of %d) written to database:' % (df.shape[0])
            for i in range(0, df.shape[0], step):
                if skip_errors:
                    try:
                        pd.io.sql.write_frame(df[i:(i + step)],
                                              con=conn,
                                              name=table_name, if_exists='append',
                                              flavor=flavor)
                        if verbose:
                            print np.where((i + step) < df.shape[0], (i + step),
                                           df.shape[0]),
                    except conn.Error, e:
                        if verbose:
                            print "Error %d: %s" % (e.args[0], e.args[1])

                else:
                    pd.io.sql.write_frame(df[i:(i + step)],
                                              con=conn,
                                              name=table_name, if_exists='append',
                                              flavor=flavor)
                    if verbose:
                        print np.where((i + step) < df.shape[0], (i + step),
                                       df.shape[0]),
            print ''

def expand_grid(x):
    if type(x) is pd.DataFrame:
        values = {col:list(x[col].unique()) for col in x}
    if type(x) is dict:
        values = {key:np.unique(value) for (key, value) in x.items()}
    
    columns = []
    lst = []
    columns += values.iterkeys()
    lst += values.itervalues()
    
    output = pd.DataFrame(list(itertools.product(*lst)), columns=columns)

    return output
