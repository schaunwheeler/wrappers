# -*- coding: utf-8 -*-

#'dialect+driver://username:password@host:port/database'
#sqlalchemy.create_engine('postgresql://schaunwheeler:@localhost:5432/herokulocaldb')

import os
import inspect
import sys
import warnings
import pandas as pd
import numpy as np
import itertools

def total_return_rate(x):
    x_min = x.min() + 0.0000000001
    return ((x.values + x_min).cumprod() - x_min)[-1]

def cagr(x):
    return ((x[-1]/x[0])**(1/len(x)))-1

def annualize(x):
    x_min = x.min() + 0.0000000001
    return (np.product(x.values+x_min)**(1/len(x)))-x_min

def maximum_drawdown_raw(x):
    return (x-pd.expanding_max(x)).min()

def maximum_drawdown_rate(x):
    x_min = x.min() + 0.0000000001
    y = ((x.values + x_min).cumprod() - x_min)
    return ((y/pd.expanding_max(y))-1).min()


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
                if diff.max() == 0.0:
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
            table_key = [df.columns[0]]

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
            step = df.shape[0]

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

def int_to_word(num,join=True):
    '''words = {} convert an integer number into words'''
    units = ['','one','two','three','four','five','six','seven','eight','nine']
    teens = ['','eleven','twelve','thirteen','fourteen','fifteen','sixteen', \
             'seventeen','eighteen','nineteen']
    tens = ['','ten','twenty','thirty','forty','fifty','sixty','seventy', \
            'eighty','ninety']
    thousands = ['','thousand','million','billion','trillion','quadrillion', \
                 'quintillion','sextillion','septillion','octillion', \
                 'nonillion','decillion','undecillion','duodecillion', \
                 'tredecillion','quattuordecillion','sexdecillion', \
                 'septendecillion','octodecillion','novemdecillion', \
                 'vigintillion']
    words = []
    if num==0: words.append('zero')
    else:
        numStr = '%d'%num
        numStrLen = len(numStr)
        groups = (numStrLen+2)//3
        numStr = numStr.zfill(groups*3)
        for i in range(0,groups*3,3):
            h,t,u = int(numStr[i]),int(numStr[i+1]),int(numStr[i+2])
            g = groups-(i//3+1)
            if h>=1:
                words.append(units[h])
                words.append('hundred')
            if t>1:
                words.append(tens[t])
                if u>=1: words.append(units[u])
            elif t==1:
                if u>=1: words.append(teens[u])
                else: words.append(tens[t])
            else:
                if u>=1: words.append(units[u])
            if (g>=1) and ((h+t+u)>0): words.append(thousands[g]+',')
    if join: return ' '.join(words)
    return words


def word_to_int(textnum, numwords={}):
    if not numwords:
      units = [
        "zero", "one", "two", "three", "four", "five", "six", "seven", "eight",
        "nine", "ten", "eleven", "twelve", "thirteen", "fourteen", "fifteen",
        "sixteen", "seventeen", "eighteen", "nineteen",
      ]

      tens = ["", "", "twenty", "thirty", "forty", "fifty", "sixty", "seventy", "eighty", "ninety"]

      scales = ["hundred", "thousand", "million", "billion", "trillion"]

      numwords["and"] = (1, 0)
      for idx, word in enumerate(units):    numwords[word] = (1, idx)
      for idx, word in enumerate(tens):     numwords[word] = (1, idx * 10)
      for idx, word in enumerate(scales):   numwords[word] = (10 ** (idx * 3 or 2), 0)

    current = result = 0
    textnum = textnum.replace(',', '')
    for word in textnum.split():
        if word not in numwords:
          raise Exception("Illegal word: " + word)

        scale, increment = numwords[word]
        current = current * scale + increment
        if scale > 100:
            result += current
            current = 0

    return result + current

def ensure_dir(f):
        d = os.path.dirname(f)
        if not os.path.exists(d):
            os.makedirs(d)

def mase(preds, original, grouping=None, func=None):
    if func is None:
        func = lambda z: np.abs(np.mean(z) - z)

    absolute_error = np.abs(preds - original)

    if grouping is not None:
        naive_ae = pd.Series(original).groupby(grouping).transform(func).values
    else:
        naive_ae = func(original)

    ase = absolute_error / naive_ae
    ase[ase==np.inf] = np.nan
    ase[ase==-np.inf] = np.nan
    ase = ase[~np.isnan(ase)]
    output = np.mean(ase<1)
    return output

def summary_covariates(df, cols, group_cols=None, sep_string='___', funcs=None):

    x = df.copy()    
    ind = x.index
    
    if (len(x.index.names)>1) | (x.index.names[0] != None):
        x = x.reset_index()

    if funcs is None:
        funcs = {'mean':np.mean}
    
    if group_cols is not None:
        y = df.groupby(group_cols)
    else:
        y = df.copy()

    for col in cols:
        for name, func in funcs.items():
            x[col+sep_string+name] = y[col].transform(lambda z: func(z))
    
    x = x.set_index(ind)
    
    return x