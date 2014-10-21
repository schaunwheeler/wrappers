# -*- coding: utf-8 -*-
"""
Created on Tue Apr 22 10:57:17 2014

@author: schaunwheeler
"""
try:
   import cPickle as pickle
except:
   import pickle
try:
    from cStringIO import StringIO
except ImportError:
    from StringIO import StringIO
import boto.s3.connection as s3
import boto.ec2.connection as ec2
from boto.s3.key import Key
import multiprocessing
import os
import subprocess
import shutil
import time
import uuid
import tempfile
import inspect
import imp
import sys
import dis
import warnings
import zlib
from cloud.serialization.cloudpickle import dumps

STORE_GLOBAL = chr(dis.opname.index('STORE_GLOBAL'))
DELETE_GLOBAL = chr(dis.opname.index('DELETE_GLOBAL'))
LOAD_GLOBAL = chr(dis.opname.index('LOAD_GLOBAL'))
GLOBAL_OPS = [STORE_GLOBAL, DELETE_GLOBAL, LOAD_GLOBAL]
HAVE_ARGUMENT = chr(dis.HAVE_ARGUMENT)
EXTENDED_ARG = chr(dis.EXTENDED_ARG)
ANACONDA_AMI = 'ami-e052b688'

class ec2do(object):
    """
    A class to manage both the mapping of arbitrary functions/inputs to EC2 
    instances, the transfer of those functions outputs to s3 buckets, and the 
    retrieval of those outputs back to a local machine. Instantiating the 
    object only requires Amazon login credentials:
    
    from awstools import ec2do
    aws_engine = ec2do(ACCESS_KEY_ID, SECRET_ACCESS_KEY)
    
    Instantiation will create an objet with the following attributes:
    
    * access_key_id: AWS access_key_id used to instantiate the object
    * secret_access_key: AWS secret_access_key used to instantiate the object
    * conn_s3: an open s3 connection using the provided login credentials
    * conn_ec2: an open ec2 connection using the provided login credentials
    * lib_dict: a dictionary of lists of python libraries. If one of the keys
    to lib_dict is fed to the ec2_call method, the method will assume that
    the module listed in the keys value are already installed in the Amazon
    machine image used to load the instance
    * reservations: an empty dictionary used to monitor running ec2 instances
    * buckets: an empty dictionary used to monitor s3 buckets
    
    """

    def __init__(self, access_key_id, secret_access_key, machine_image, 
                 **ec2_kwargs):
        self.access_key_id = access_key_id
        self.secret_access_key = secret_access_key
        self.machine_image = machine_image
        self.conn_s3 = s3.S3Connection(access_key_id, secret_access_key)
        self.conn_ec2 = ec2.EC2Connection(access_key_id, secret_access_key, 
            **ec2_kwargs)
        self.lib_dict = {'anaconda': ['apptools', 'argcomplete', 'astropy', 
            'atom', 'beautiful-soup', 'binstar', 'biopython', 'bitarray', 
            'blaze', 'blz', 'bokeh', 'boto', 'cairo', 'casuarius', 'cdecimal', 
            'chaco', 'colorama', 'conda', 'conda-build', 'configobj', 'cubes', 
            'curl', 'cython', 'datashape', 'dateutil', 'disco', 'docutils', 
            'dynd-python', 'enable', 'enaml', 'envisage', 'erlang', 'flask', 
            'freetype', 'future', 'gevent', 'gevent-websocket', 
            'gevent_zeromq', 'greenlet', 'grin', 'h5py', 'hdf5', 'ipython', 
            'itsdangerous', 'jinja2', 'keyring', 'kiwisolver', 'launcher', 
            'libnetcdf', 'libpng', 'libsodium', 'libtiff', 'libxml2', 
            'libxslt', 'llvm', 'llvmpy', 'lxml', 'markupsafe', 'matplotlib', 
            'mayavi', 'mdp', 'menuinst', 'mingw', 'mock', 'mpi4py', 'mpich2', 
            'netcdf4', 'networkx', 'nltk', 'nose', 'numba', 'numexpr', 'numpy', 
            'opencv', 'openpyxl', 'openssl', 'pandas', 'patsy', 'pep8', 'pil', 
            'python-pip', 'ply', 'psutil', 'py', 'py2cairo', 'pyaudio', 'pycosat', 
            'pycparser', 'pycrypto', 'pycurl', 'pyface', 'pyflakes', 
            'pygments', 'pykit', 'pyparsing', 'pyreadline', 'pysal', 'pysam', 
            'pyside', 'pytables', 'pytest', 'python', 'pytz', 'pywin32', 
            'pyyaml', 'pyzmq', 'qt', 'redis', 'redis-py', 'requests', 'rope', 
            'scikit-image', 'scikit-learn', 'scipy', 'setuptools', 'six', 
            'sphinx', 'spyder', 'sqlalchemy', 'ssl_match_hostname', 
            'statsmodels', 'sympy', 'theano', 'tk', 'tornado', 'traits', 
            'traitsui', 'ujson', 'vtk', 'werkzeug', 'xlrd', 'xlsxwriter', 
            'xlwt', 'yaml', 'zeromq', 'zlib', 'sklearn']}
        self.reservations = {}
        self.scriptkeys = {}
    
    def to_s3(self, obj, bucket_id, key, protocol=0, **kwargs):
        """
        Save object to Amazon S3 as string representation
        
        Parameters:
        obj: the Pandas object to be saved
        conn: an S3 connection or S3 bucket object
        bucket_id: the name of the bucket where the object is located
        key: the key to assign to the saved object
        protocol: passed to pickle.dumps
        **kwargs: passed to pandas boto `set_contents_from_string`
        
        """
        
        if type(obj)==str:
            obj_string = zlib.compress(obj)
        else:
            try:
                obj_string = zlib.compress(dumps(obj, protocol))
            except:
                raise Exception('obj could not be pickled')
        
        try:
            bucket = self.conn_s3.get_bucket(bucket_id)
        except:
            bucket = self.conn_s3.create_bucket(bucket_id)
            
        k = Key(bucket)
        k.key = key
        _ = k.set_contents_from_string(obj_string, **kwargs)
    
        return key
    
    def from_s3(self, bucket_id, key):
        """
        Load object from Amazon S3
        
        Parameters:
        conn: an S3 connection or S3 bucket object
        bucket_id: the name of the bucket where the object is located
        key: the key assigned to the saved object
        protocol: passed to pickle.loads
        
        """
    
        bucket  = self.conn_s3.get_bucket(bucket_id)        
            
        obj_string = bucket.get_key(key).read()
        
        obj_string = zlib.decompress(obj_string)        
        
        try:
            return pickle.loads(obj_string)
        except:
            return obj_string

    @staticmethod
    def quotes(s, remove=True):
        empty = chr(32)[:0]
        double = [100, 111, 117, 98, 108, 101]
        middle = [95, 95, 95, 113, 95, 95, 95]
        single = [115, 105, 110, 103, 108, 101]
        grave = [103, 114, 97, 118, 101]
        double_r = empty.join([chr(x) for x in double+middle+double])
        single_r = empty.join([chr(x) for x in single+middle+single])
        grave_r = empty.join([chr(x) for x in grave+middle+grave])
        if remove:
            return s.replace(chr(34), double_r).replace(chr(39), single_r
                ).replace(chr(96), grave_r)
        else:
            return s.replace(double_r, chr(34)).replace(single_r, chr(39)
                ).replace(grave_r, chr(96))

    @staticmethod
    def manual_module(path, name, from_string=True):
        new_module = imp.new_module(name)
        if not from_string:
            with open(path, 'r') as f:
                new_module_code = f.read()
        else:
            new_module_code = path
        new_module_code = new_module_code
        exec new_module_code in new_module.__dict__
        sys.modules[name] = new_module

    @staticmethod    
    def flag_objects(x):
        '''Flags whether an object is a module, class, or function'''
        return inspect.ismodule(x) | inspect.isclass(x) | inspect.isfunction(x)

    @staticmethod    
    def flag_interactive_objects(x):
        '''Flags whether an object is an interactively-defined class or function'''
        test = (inspect.isclass(x) | inspect.isfunction(x))
        if test:
            return x.__module__=='__main__'
        else:
            return False
        
    def extract_code_dependencies(self, func):
        """
        Find all global modules, classes, or functions read or written to by a 
        function
        """
        
        co =func.func_code
        code = co.co_code
        names = co.co_names
        allglobal = func.func_globals
        allglobal = {x:y for x,y in allglobal.items() if self.flag_objects(y)}
        out_names = set()
    
        n = len(code)
        i = 0
        extended_arg = 0
        while i < n:
            op = code[i]
    
            i = i+1
            if op >= HAVE_ARGUMENT:
                oparg = ord(code[i]) + ord(code[i+1])*256 + extended_arg
                extended_arg = 0
                i = i+2
                if op == EXTENDED_ARG:
                    extended_arg = oparg*65536L
                if op in GLOBAL_OPS:
                    out_names.add(names[oparg])
                    
        output = {x:y for x,y in allglobal.items() if 
            (x in out_names) | (y.__name__ in out_names)}
            
        return output
    
    def get_objects(self, func, mask='anaconda'):
        """
        Return a class with three attributes:
        
        * imports: python modules that need to be imported for a function to be
        run on EC2
        * installs: python modules that will first need to be installed (via
        apt-get) in order for a function to be run on EC2
        * full_files: local paths to scripts that will need to be imported
        as custom modules in order for a function to be run on EC2
        
        """
        imports_list = self.extract_code_dependencies(func)
        inter = [k for k,v in imports_list.items() if 
            self.flag_interactive_objects(v)]
        del v
        n_inter = len(inter)
        new_n_inter = n_inter*2
        while new_n_inter>n_inter:
            for k in inter:
                new_imports = self.extract_code_dependencies(imports_list[k])
                imports_list.update(new_imports)
                inter = [k for k,v in imports_list.items() if 
                    self.flag_interactive_objects(v)]
                del v
                n_inter = new_n_inter
                new_n_inter = len(inter)

        try:
            parent_module = func.__module__
        except:
            parent_module = '__main__'

        not_in_keys = parent_module not in imports_list.keys()
        not_in_main = parent_module!='__main__'

        if not_in_keys & not_in_main:
            imports_list[parent_module] = sys.modules[parent_module]
            
        line_items = []
        apt_get = []
        load_files = []
        for alias, obj in imports_list.items():
            if inspect.ismodule(obj):
                line_item = 'import %s' % obj.__name__
                root_module = obj.__file__.split('/')
            elif obj.__module__ != '__main__':
                line_item = 'from %s import %s' % (obj.__module__,
                                                    obj.__name__)
                root_module = inspect.getsourcefile(obj).split('/')
            else:
                pickled_obj = dumps(obj)
                line_item = '%s = pickle.loads(quotes(%s, remove=False))\n' % (
                    alias, repr(self.quotes(pickled_obj)))
                root_module = None
    
            if obj.__name__ != alias:
                line_item += ' as %s' % alias
    
            ind = None

            if root_module is not None:    
                try:
                    ind = root_module.index('site-packages')
                except:
                    try:
                        _ = root_module.index(
                            'python%d.%d' % (sys.version_info[0:2]))
                    except:
                        source_file = inspect.getsourcefile(obj)
                        load_files.append({'name':alias, 'path':source_file})
    
                if ind is not None:
                    root_module = root_module[ind+1]
                    apt_get.append(root_module)
    
            line_items.append(line_item)
        
        if mask is not None:
            apt_get = [item for item in apt_get if not 
                any([item.startswith(x) for x in self.lib_dict[mask]])]
        
        class script_setup(object):
            imports =  list(set(line_items))
            installs = list(set(apt_get))
            full_files = load_files
    
        return script_setup

    def prep_buckets(self):
        script_bucket_name = str(uuid.uuid4())
        collection_bucket_name = str(uuid.uuid4())
        
        try:
            script_bucket = self.conn_s3.get_bucket(script_bucket_name)
        except:
            script_bucket = self.conn_s3.create_bucket(script_bucket_name)
        
        try:
            collection_bucket = self.conn_s3.get_bucket(collection_bucket_name)
        except:
            collection_bucket = self.conn_s3.create_bucket(collection_bucket_name)

        self.script_bucket = script_bucket
        self.collection_bucket = collection_bucket
    
    def create_script(self, func, func_kwargs=None, mask='anaconda', 
        aptget=None, custom=None):
        """
        Create a custom python script to run a function on EC2.
        
        Parameters
        __________
        
        * func: an arbitrary function
        * bucket_id: the name of the s3 bucket to which to load func results
        * func_kwargs: a dictionary of keyword arguments to feed to func
        * mask: a key in lib_dicts indicating which Python modules should be
        assumed loaded on EC2
        * apt_get: a list of package names to install on EC2 via apt-get (not 
        tested)
        * custom: a custom script to run in the EC2 shell before starting 
        python (not tested)
        
        """
        
        # get function dependencies
        setup_specs = self.get_objects(func, mask=mask)
    
        # start script
        script = '#!/usr/bin/env python\n'
    
        # always install these modules
        script += 'try:\n    import cPickle as pickle\n'
        script += 'except:\n    import pickle\n\n'
        script += 'try:\n    from cStringIO import StringIO\n'
        script += 'except:\n    from StringIO import StringIO\n\n'
        script += 'import sys, os, time, uuid, inspect, imp, dis, subprocess, zlib\n\n'
        script += 'from cloud.serialization.cloudpickle import dumps\n'
        script += 'import boto.s3.connection as s3\n'
        script += 'from boto.s3.key import Key\n'
        script += 'def quotes(s, remove=True):\n'
        script += '    empty = chr(32)[:0]\n'
        script += '    double = [100, 111, 117, 98, 108, 101]\n'
        script += '    middle = [95, 95, 95, 113, 95, 95, 95]\n'
        script += '    single = [115, 105, 110, 103, 108, 101]\n'
        script += '    grave = [103, 114, 97, 118, 101]\n'
        script += '    double_r = empty.join([chr(x) for x in double+middle+double])\n'
        script += '    single_r = empty.join([chr(x) for x in single+middle+single])\n'
        script += '    grave_r = empty.join([chr(x) for x in grave+middle+grave])\n'
        script += '    if remove:\n'
        script += '        return s.replace(chr(34), double_r).replace(chr(39), single_r\n'
        script += '            ).replace(chr(96), grave_r)\n'
        script += '    else:\n'
        script += '        return s.replace(double_r, chr(34)).replace(single_r, chr(39)\n'
        script += '            ).replace(grave_r, chr(96))\n\n'
        script += 'def manual_module(path, name, from_string=True):\n'
        script += '    new_module = imp.new_module(name)\n'
        script += '    if not from_string:\n'
        script += '        with open(path, \'r\') as f:\n'
        script += '            new_module_code = f.read()\n'
        script += '    else:\n'
        script += '        new_module_code = path\n'
        script += '    new_module_code = new_module_code\n'
        script += '    exec new_module_code in new_module.__dict__\n'
        script += '    sys.modules[name] = new_module\n\n'
        script += 'def to_s3(obj, conn, bucket_id, key, **kwargs):\n'
        script += '    if type(obj)==str:\n'
        script += '        obj_string = zlib.compress(obj)\n'
        script += '    else:\n'
        script += '        try:\n'
        script += '            obj_string = zlib.compress(dumps(obj))\n'
        script += '        except:\n'
        script += '            raise Exception(\'obj could not be pickled\')\n'
        script += '    if type(conn) is s3.S3Connection:\n'
        script += '        try:\n'
        script += '            bucket = conn.get_bucket(bucket_id)\n'
        script += '        except:\n'
        script += '            bucket = conn.create_bucket(bucket_id)\n'
        script += '    elif type(conn) is s3.bucket.Bucket:\n'
        script += '        bucket = conn\n'
        script += '    k = Key(bucket)\n'
        script += '    k.key = key\n'
        script += '    _ = k.set_contents_from_string(obj_string, **kwargs)\n'
        script += '    return key\n\n'

        # establish an s3 connection
        script += "conn_s3 = s3.S3Connection('%(aki)s', '%(sak)s')\n" % {
            'aki': self.access_key_id, 'sak': self.secret_access_key}
        
        # install any non-python programs necessary
        if aptget is not None:
            script += 'subprocess.call([%s, %s])\n' % (repr('apt-get'), 
                repr('upgrade'))
            for a in aptget:
                script += 'subprocess.call([%s, %s, %s, %s])\n' % (
                    repr('apt-get'), repr('install'), repr('-y'), repr(a))
    
        # install pip then install modules
        if len(setup_specs.installs) > 0:
            if mask is not None:
                if 'python-pip' not in mask:
                    script += 'subprocess.call([%s, %s, %s])\n' % (
                        repr('apt-get'), repr('install'), repr('python-pip'))
            for p in setup_specs.installs:
                script += 'subprocess.call([%s, %s, %s, %s])\n' % (
                    repr('pip'), repr('install'), repr('-y'), repr(p))

        # do anything else before opening the python interpreter
        if custom is not None:
            for c in custom:
                script += 'subprocess.call([%s])\n' % repr(c)
                
        # import custom modules (not on path)
        if len(setup_specs.full_files)>0:
            for item in setup_specs.full_files:
                module_name = item['name']
                with open(item['path'], 'r') as f:
                    module_code = f.read()
                module_code = self.quotes(module_code)
                script += 'manual_module(quotes(%s, remove=False), %s, from_string=True)\n' % (
                    repr(module_code), repr(module_name))
                script += 'import %s\n' % module_name
                script += 'from %s import *\n' % module_name
        
        # import any other modules necessary for the function to run
        if len(setup_specs.imports)>0:
            for i in setup_specs.imports:
                script += '%s\n' % i
    
        # recreate and run the function
        dumped_code = dumps(func)
        dumped_code = self.quotes(dumped_code)
        script += 'do_func_code = quotes(%s, remove=False)\n' % repr(dumped_code)
        script += 'do_func = pickle.loads(do_func_code)\n'
        script += 'f_glob = do_func.func_globals\n'
        script += 'exports = [x for x in f_glob if not x.startswith(chr(95)*2)]\n'
        script += 'for name in exports:\n'
        script += '    globals()[name]  = f_glob[name]\n'
    
        if func_kwargs is None:
            script += 'output = do_func()\n'
        else:
            script += 'kwargs = quotes(%s, remove=False)\n' % (
                repr(self.quotes(dumps(func_kwargs))))
            script += 'kwargs = pickle.loads(kwargs)\n'
        
            script += 'output = do_func(**kwargs)\n'
    
        # transfer results to s3
        script += 'key_id = str(uuid.uuid1())\n'
        script += 'to_s3(output, conn=conn_s3, bucket_id=%s, key=key_id)\n' % (
            repr(self.collection_bucket.name))
        #script += 'print %s\n' % repr('working script completed')
        return script.strip('\n')

    def local_eval(self, func, func_kwargs=None,
        aptget=None, custom=None, mask='anaconda'):
        """
        Build a script and run it on the local machine (for testing). Return
        the bucket object to which results were sent.
        
        """
    
        work_script = self.create_script(func=func, func_kwargs=func_kwargs, 
            aptget=aptget, custom=custom, mask='anaconda')
    
        temp_dir = tempfile.mkdtemp()
        temp_file = os.path.join(temp_dir, 'work_script.py')
        with open(temp_file, "w") as f:
            f.write(work_script)
        subprocess.call(['python', temp_file])
        shutil.rmtree(temp_dir)
        
        return self.collection_bucket

    def load_script(self, script):
        ws_key = str(uuid.uuid4())

        _ = self.to_s3(script, bucket_id=self.script_bucket.name, key=ws_key)
        
        key = self.script_bucket.get_key(ws_key)
        key.set_acl('public-read')   
        return ws_key

    def ec2_eval(self, script_key, python_dir='/home/ubuntu/anaconda/bin',
        n_instances=1, keypair=None, auto_shutdown=True, **kwargs):

        """
        Build a script and run in on EC2.
        
        Parameters
        __________
        
        * func: an arbitrary function
        * python_dir: the full path to python on the EC2 instance(s)
        * func_kwargs: a dictionary of keyword arguments to feed to func
        * temp_bucket: the name of the bucket to use to store the script and 
        collect results
        * n_instances: the number of instances on which to run func
        * instance_type: the type of instance(s) to start
        * keypair: if given, the name of the keypair to use to ssh into the 
        instance(s)
        * aptget: pass to create_script
        * custom: pass to create_script
        * mask: pass to create_script
        * auto_shutdown: boolean, whether to include a command to automatically
        shut down the instance(s) when the pyhton script finishes running
        
        """
                                        
        key_url = 'http://{host}/{bucket}/{key}'.format(
            host=self.conn_s3.server_name(), bucket=self.script_bucket.name, 
            key=script_key)
        
        bash_script = '#!/bin/bash\n'
        bash_script += 'wget %s -O work_script\n' % key_url
        bash_script += 'openssl zlib -d < work_script > work_script.py\n'
        bash_script += os.path.join(python_dir, 'python work_script.py\n')
        if auto_shutdown:
            bash_script += '/sbin/shutdown now -h\n'
                
        if keypair is None:
            temp_key = 'keypair_'+str(uuid.uuid1())
            key_pair = self.conn_ec2.create_key_pair(temp_key)
            key_pair_name = key_pair.name
        else:
            key_pair = self.conn_ec2.get_key_pair(keypair)
            key_pair_name = keypair
        
        reservation = self.conn_ec2.run_instances(
            image_id=self.machine_image, key_name=key_pair_name,
            min_count=n_instances, max_count=n_instances,
            user_data=bash_script, **kwargs)
                
        self.reservations[reservation.id] = reservation
        self.scriptkeys[reservation.id] = script_key

    def poll_instances(self):
        """return True if all instances have terminated"""
        
        inst = [x.instances for x in self.reservations.values()]
        inst = [x for y in inst for x in y]
        n_inst = len(inst)
        statuses = [x.update() for x in inst]
        status = n_inst - sum([x=='terminated' for x in statuses])
        return status

    def poll_outputs(self):
        """return True if all instances have an output in s3"""
        
        inst = [x.instances for x in self.reservations.values()]
        inst = [x for y in inst for x in y]
        n_inst = len(inst)
        n_keys = len(self.collection_bucket.get_all_keys())
        return n_inst-n_keys

    def get_outputs(self, force=False):
        if force:
            if self.poll_outputs() > 0:
                warnings.warn('not all instances have outputed to s3')
            elif self.poll_instances() > 0:
                warnings.warn('not all instances have terminated')
        else:
            if self.poll_outputs() > 0:
                raise Exception('not all instances have outputed to s3')
            elif self.poll_instances() > 0:
                raise Exception('not all instances have terminated')
            
        keys = self.collection_bucket.get_all_keys()
        outputs = [self.from_s3(self.collection_bucket.name, x) for x in keys]
        _ = self.collection_bucket.delete_keys([x.name for x in keys])
        _ = self.conn_s3.delete_bucket(self.collection_bucket.name)

        keys = self.script_bucket.get_all_keys()
        _ = self.script_bucket.delete_keys([x.name for x in keys])
        _ = self.conn_s3.delete_bucket(self.script_bucket.name)

        _ = self.reservations = None
        _ = self.scriptkeys = None

        return outputs
    
    @staticmethod
    def get_time_lapse(ref):
        t = time.time()                        
        m, s = divmod(t-ref, 60)
        h, m = divmod(m, 60)
        return "%d:%02d:%02d" % (h, m, s)            
            
    def monitor_run(self, verbose=True, sleep=5, ref_time=None):        
        if ref_time is None:
            t0 = time.time()
        else:
            t0 = ref_time

        i = self.poll_instances()
        o = self.poll_outputs()
    
        if verbose:
            print '\nwaiting:  outputs  |  instances  ||  time lapsed'
            print '%s  |  %s  ||  %s' % (str(o).rjust(17), str(i).rjust(9), 
                                         self.get_time_lapse(t0).rjust(11))
    
        while (o >0) | (i>0):
            i_new = self.poll_instances()
            o_new = self.poll_outputs()
            if (o>o_new) | (i>i_new):
                if verbose:
                    print '%s  |  %s  ||  %s' % (str(o_new).rjust(17), 
                        str(i_new).rjust(9), self.get_time_lapse(t0).rjust(11))
                o = o_new
                i = i_new
            if i < o:
                raise Exception('instance terminated wtihout yielding output')
            time.sleep(sleep)

    @staticmethod
    def subset_kwargs(func, kwargs, explicit=[]):
        func_args = inspect.getargspec(func).args
        func_args = [x for x in func_args if x not in explicit]
        new_kwargs = {}
        for kw in func_args:
            if kwargs.has_key(kw):
                new_kwargs[kw] = kwargs.pop(kw, None)
        return new_kwargs

    def map(self, func, input_list, verbose=True, **kwargs):
        t0 = time.time()
        n_inputs = len(input_list)
        
        self.prep_buckets()
        cs_kwargs = self.subset_kwargs(self.create_script, kwargs, 
            explicit=['self', 'func', 'func_kwargs'])
        
        try:
            single = input_list[1:] == input_list[:-1]
        except:
            single = False
        
        if single:
            if verbose:
                print 'creating script...'
            work_script = self.create_script(func=func, 
                func_kwargs=input_list[0], **cs_kwargs)
            if verbose:
                print 'loading script...'
            ws_key = self.load_script(work_script)
            key_list = [ws_key]*n_inputs
        else:
            key_list = []
            if verbose:
                print 'creating and loading scripts...'
                inc = 0
            for item in input_list:
                work_script = self.create_script(func=func, func_kwargs=item, 
                    **cs_kwargs)
                ws_key = self.load_script(work_script)
                key_list.append(ws_key)
                if verbose:
                    inc += 1
                    print inc,

        if verbose:
            print '\nstarting instances...'
            inc = 0
        for k in key_list:
            self.ec2_eval(script_key=k, **kwargs)
            if verbose:
                inc += 1
                print inc,
        _ = self.monitor_run(verbose=verbose, ref_time=t0)
        
        if verbose:
            print '\ncollecting results...'

        outputs = self.get_outputs()

        return outputs


def cleanup():
    uuid_regex = '%s{8}-%s{4}-%s{4}-%s{4}-%s{12}' % (('[A-Za-z0-9]',) * 5)
    return uuid_regex

# Possible convenience functon for comparing AWS performance to others
def interchange(func, input_list, engine=None, **awskwargs):

    if engine=='multiprocessing':
        engine = multiprocessing.pool.Pool()
        output = [engine.apply_async(func, x) for x in input_list]
        engine.close()
        engine.join()
        output = [o.get() for o in output] 
    elif engine is not None:
        output = engine.map(func=func, input_list=input_list, **awskwargs)
    else:
        output = [func(x) for x in input_list]
    
    return output

def estimate_price(runtime):
    types = [u'm3.medium', u'c3.large', u'c3.xlarge', u'c3.2xlarge',
              u'c3.4xlarge',u'c3.8xlarge']
    rates = [0.070, 0.105, 0.210, 0.420, 0.840, 1.680]
    cores = [1, 2, 4, 8, 16, 32]
    
    for i in range(len(rates)):
        total_time = (runtime/cores[i])
        total_time = total_time + 2
        total_time = total_time/60
        total_cost = total_time * rates[i] * 500
        print '%s: %2d core - $%1.2f - %3.1f minutes' % (
            types[i].rjust(10), cores[i], total_cost, total_time*60)