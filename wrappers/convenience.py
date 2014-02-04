# -*- coding: utf-8 -*-

import os
import inspect
import sys

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
