# -*- coding: utf-8 -*-
"""
Created on Thu Aug  7 09:22:34 2014

@author: schaunwheeler
"""

import pandas as pd
import numpy as np
import colorsys
from matplotlib.colors import ColorConverter
import tempfile
import re
import uuid
import bokeh
import bokeh.plotting
import bokeh.colors
import os
from collections import OrderedDict
from itertools import izip_longest

DSTYPE = bokeh.objects.ColumnDataSource

HTML_HEAD = '''
<!DOCTYPE html>
<html>
<head>
        
<link rel="stylesheet" href="http://maxcdn.bootstrapcdn.com/bootstrap/3.2.0/css/bootstrap.min.css">
<link rel="stylesheet" href="http://maxcdn.bootstrapcdn.com/bootstrap/3.2.0/css/bootstrap-theme.min.css">
<link rel="stylesheet" href="http://cdn.pydata.org/bokeh-0.6.0.css" type="text/css" />
<link rel="stylesheet" href="http://cdn.datatables.net/1.10.2/css/jquery.dataTables.min.css">

<script src="http://ajax.googleapis.com/ajax/libs/jquery/2.1.1/jquery.min.js"></script>
<script src="http://maxcdn.bootstrapcdn.com/bootstrap/3.2.0/js/bootstrap.min.js"></script>
<script type="text/javascript" src="http://cdn.pydata.org/bokeh-0.6.0.js"></script>
<script type="text/javascript" src="http://cdn.datatables.net/1.10.2/js/jquery.dataTables.min.js"></script>
'''

HTML_SCRIPTS = '''
<script type="text/javascript">

function get_plot(id, folder, target) {
    var s = document.createElement("script");
    s.type = "text/javascript";
    s.src = folder.concat(id).concat(".js");
    s.id = id;
    s.async = "true";
    s.setAttribute("data-bokeh-data", "static");
    s.setAttribute("data-bokeh-modelid", id);
    s.setAttribute("data-bokeh-modeltype", "Plot");
    var elem = $('#'+target).find('.plotholder').get(0);
    if (elem.children.length > 0) {
        $(elem).empty();
    }
    elem.appendChild(s);
}

function get_table(id, folder, target) {
    var s = document.createElement("script");
    s.type = "text/javascript";
    s.src = folder.concat(id).concat("_table.js");
    s.id = id+'_table';
    s.async = "true";
    var elem = $('#'+target).find('.tableholder').get(0);
    if (elem.children.length > 0) {
        $(elem).empty();
    }
    elem.appendChild(s);

}

function wait(name, callback) {
    var interval = 10; // ms
    if (typeof name=="string") {
        window.setTimeout(function() {
            if ($(name).length>0) {
                callback();
            } else {
                window.setTimeout(arguments.callee, interval);
            }
        }, interval);
    } else {
        window.setTimeout(function() {
            if (name) {
                callback();
            } else {
                window.setTimeout(arguments.callee, interval);
            }
        }, interval);
    }

}

window.onload = function () {

     $("a[data-file]").on('click', function(){
         var filename = this.getAttribute('data-file');
         var foldername = this.getAttribute('data-folder');
         var targetname = this.getAttribute('data-target');
         get_plot(filename, foldername, targetname)
         wait(".bk-logo", function(){
            $(".bk-logo")[0].setAttribute("href", "http://www.infusiveintelligence.com/");
            $(".bk-toolbar-button.help").remove();
            var elements = document.getElementsByTagName('a');
            for (var i = 0, len = elements.length; i < len; i++) {
                elements[i].removeAttribute('title');
            }
        });
        var table = $("#"+targetname+" .tableholder")
        if (table.length>0) {
            get_table(filename, foldername, targetname);
        }
    });

    var anchors = $("a[data-file]");
    var items = {};
    $(anchors).each(function() {
        items[$(this).attr('data-target')] = true;
    });

    for(var target in items) {
        $("a[data-target='"+target+"']")[0].click();
    }

    $('#nav-expander').on('click',function(e){
      e.preventDefault();
      $('body').toggleClass('nav-expanded');
    });

    $('.sub-nav').on('click',function(e){
      e.preventDefault();
      $('body').removeClass('nav-expanded');
    });

    $('.main-nav').on('click',function(e){
      e.preventDefault();
      $(this).siblings().find('.sub-nav').addClass('hide')
      $(this).find('.sub-nav').toggleClass('hide')
    });
}

</script>
'''

HTML_CSS = '''
<style type="text/css">
body {
     font-family: "Century Gothic", CenturyGothic, AppleGothic, sans-serif;
}

.container-fluid {
  margin-top: 60px;
}

#footer {
    clear:both;
    width=:100%;
    color: #808080;
    padding: 10px;
}

hr {
    border: 0;
    height: 0;
    border-top: 1px solid rgba(0, 0, 0, 0.1);
    border-bottom: 1px solid rgba(255, 255, 255, 0.3);
    margin-top: 0px;
}

#footer hr {
  padding: 0px;
  margin: 0px;
}

a.nav-expander {
  background: none repeat scroll 0 0 #000000;
  color: #FFFFFF;
  display: block;
  font-size: 15px;
  font-weight: 400;
  height: 50px;
  margin-right: 0;
  padding: 1em 1.6em 2em;
  position: absolute;
  right: 0;
  text-decoration: none;
  text-transform: uppercase;
  top: 0;
  transition: right 0.3s ease-in-out 0s;
  width: 100px;
  z-index: 12;
  transition: right 0.3s ease-in-out 0s;
  -webkit-transition: right 0.3s ease-in-out 0s;
  -moz-transition: right 0.3s ease-in-out 0s;
  -o-transition: right 0.3s ease-in-out 0s;

}
a.nav-expander:hover {
  cursor: pointer;
}
a.nav-expander.fixed {
  position: fixed;
}
.nav-expanded a.nav-expander.fixed {
    right: 20em;
}
.sub-nav {
    padding-left: 10px;
}
.sub-nav > a {
    color: black;
}
.main-nav {
  padding-left: 20px;
}
.main-nav > a {
    color: black;
}
.main-nav > a:after {
    content: "";
    display: block;
    height: 1px;
    margin: 0px;
    background: black;
    margin-bottom: 10px;
    margin-left: -20px;
}
nav {
  background: #C6C6C6;
  display: block;
  height: 100%;
  overflow: auto;
  position: fixed;
  right: -20em;
  font-size: 15px;
  top: 0;
  width: 20em;
  z-index: 2000;
  transition: right 0.3s ease-in-out 0s;
  -webkit-transition: right 0.3s ease-in-out 0s;
  -moz-transition: right 0.3s ease-in-out 0s;
  -o-transition: right 0.3s ease-in-out 0s;

}
.nav-expanded nav {
  right: 0;
}
body.nav-expanded {
  margin-left: 0em;
  transition: right 0.4s ease-in-out 0s;
  -webkit-transition: right 0.4s ease-in-out 0s;
  -moz-transition: right 0.4s ease-in-out 0s;
  -o-transition: right 0.4s ease-in-out 0s;
}
#nav-close {
  font-weight: 300;
  font-size: 24px;
  padding-right: 10px;
}
.tableholder {
  overflow:scroll;
}

TD {
    font-size: 10px;
    font-family: "Century Gothic", CenturyGothic, AppleGothic, sans-serif;
}

.dataTables_wrapper .dataTables_paginate .paginate_button.current {
    color:white !important;
    border-radius: 3px !important;
    border:0px solid #cacaca;
    background-color:#A8A8A8;
    background:#A8A8A8;
    }

.dataTables_wrapper .dataTables_paginate .paginate_button.current:hover {
    color:white !important;
    border-radius: 3px !important;
    border:0px solid #cacaca;
    background-color:#C6C6C6;
    background:#C6C6C6;
    }

.dataTables_wrapper .dataTables_paginate .paginate_button:hover {
    color:white !important;
    border-radius: 3px !important;
    border:0px solid #cacaca;
    background-color:#C6C6C6;
    background:#C6C6C6;
    }

table.dataTable.compact thead th, table.dataTable.compact thead td {
    padding: 5px 2px;
}

::-webkit-input-placeholder {
   color: white;
   font-size: 11px;
}
:-moz-placeholder { /* Firefox 18- */
   color: white;
   font-size: 11px;
}
::-moz-placeholder {  /* Firefox 19+ */
   color: white;
   font-size: 11px;
}
:-ms-input-placeholder {
   color: white;
   font-size: 11px;
}

table.dataTable thead .sorting,
table.dataTable thead .sorting_asc,
table.dataTable thead .sorting_desc {
    background : none;
}

button, input, optgroup, textarea {
  background: #C6C6C6;
  border: transparent;
  color: black;
  font-weight: normal;
  border-radius: 3px !important;
  padding: 3px;
}

select {
  background: #C6C6C6;
  border: transparent;
  color: white;
  font-weight: normal;
  border-radius: 3px !important;
  padding: 3px;
}

thead input {
  width: 100%;
  padding: 3px;
  box-sizing: border-box;
  font-size: 11px;
}

label {
  font-weight: normal;
  font-size: 11px;
  margin-bottom: 0px;
}

.dataTables_wrapper .dataTables_length, .dataTables_wrapper .dataTables_filter, .dataTables_wrapper .dataTables_info,
.dataTables_wrapper .dataTables_processing, .dataTables_wrapper .dataTables_paginate {
  font-size: 11px;
}

.bk-logo-medium {
  width: 35px;
  height: 35px;
  background-image: url(data:image/png;base64,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);
}

.bk-logo-small {
    width: 26px;
    height: 26px;
    background-image: url(data:image/png;base64,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);
}

.bk-toolbar-button > span.tip {
  transition: all 0.3s ease;
  -webkit-transition: all 0.3s ease;
  -moz-transition: all 0.3s ease;
  -o-transition: all 0.3s ease;
  font-family: "Century Gothic", CenturyGothic, AppleGothic, sans-serif;
}

.bokeh_tooltip.left::before {
  border-color: transparent #808080 transparent transparent;
  }
.bokeh_tooltip.right::after {
  border-color: transparent transparent transparent #808080;
 }
.bokeh_tooltip {
  padding: 10px;
  background-color: #808080;
  border-radius: 20px;
}
.bokeh_tooltip_row_label {
  color: black;
  font-family: "Century Gothic", CenturyGothic, AppleGothic, sans-serif;
}
.bokeh_tooltip_row_value {
  color: white;
  font-family: "Century Gothic", CenturyGothic, AppleGothic, sans-serif;
  padding-left: 5%;
}

</style>
'''


def get_nodes(links, node):
    d = {}
    d['name'] = node
    children = [x[1] for x in links if x[0] == node]
    if children:
        d['children'] = [get_nodes(links, child) for child in children]
    return d


class DefaultOrderedDict(OrderedDict):
    # taken from http://stackoverflow.com/a/4127426/1829950
    def __init__(self, *args, **kwargs):
        if not args:
            self.default_factory = None
        else:
            if not (args[0] is None or callable(args[0])):
                raise TypeError('first argument must be callable or None')
            self.default_factory = args[0]
            args = args[1:]
        super(DefaultOrderedDict, self).__init__(*args, **kwargs)

    def __missing__(self, key):
        if self.default_factory is None:
            raise KeyError(key)
        self[key] = default = self.default_factory()
        return default

    def __reduce__(self):  # optional, for pickle support
        args = (self.default_factory,) if self.default_factory else ()
        return self.__class__, args, None, None, self.iteritems()


def eigsorted(cov):
    vals, vecs = np.linalg.eigh(cov)
    order = vals.argsort()[::-1]
    return vals[order], vecs[:, order]


def calculate_ovals(x, y, sd=3):

    cov = np.cov(x, y)
    lambda_, v = eigsorted(cov)
    lambda_ = np.sqrt(lambda_)
    
    x = np.mean(x)
    y = np.mean(y)
    width = lambda_[0]*sd*2
    height = lambda_[1]*sd*2
    angle = np.degrees(np.arctan2(*v[:, 0][::-1]))

    out = pd.Series({
        'x': x, 'y': y,
        'width': width, 'height': height,
        'angle': angle}).astype(float)
    return out    


def change_axes(**kwargs):

    p = bokeh.plotting.curplot() 
    if p is None: 
        return None

    x_axis = bokeh.plotting.xaxis()
    y_axis = bokeh.plotting.yaxis()

    xtags = sum([1 if k.startswith('x_') else 0 for k in kwargs.keys()])
    ytags = sum([1 if k.startswith('y_') else 0 for k in kwargs.keys()])
    
    if xtags > 0:
        xs = [(k[2:], v) for k, v in kwargs.items() if k.startswith('x_')]
        for k, v in xs:
            for i in range(len(x_axis)):
                x_axis[i].__setattr__(k, v)
    if ytags > 0:
        ys = [(k[2:], v) for k, v in kwargs.items() if k.startswith('y_')]
        for k, v in ys:
            for i in range(len(y_axis)):
                y_axis[i].__setattr__(k, v)

    if (xtags + ytags) < len(kwargs):
        boths = [(k, v) for k, v in kwargs.items() if (k.startswith('x_')+k.startswith('y_')) == 0]
        for k, v in boths:
            for i in range(len(x_axis)):
                x_axis[i].__setattr__(k, v)
            for i in range(len(y_axis)):
                y_axis[i].__setattr__(k, v)

    bind = x_axis+y_axis
    return bind


def change_legend(**kwargs):
    legend = bokeh.plotting.legend()[0]
    for k, v in kwargs.items():
        legend.__setattr__(k, v)
    return legend


def set_hover(values):
    hover = [t for t in bokeh.plotting.curplot().tools if isinstance(t, bokeh.objects.HoverTool)][0]
    hover.tooltips = OrderedDict(values)
    return hover


def facet_wrap(plots, ncol, name=None):
    args = [iter(plots)] * ncol
    plot_list = izip_longest(fillvalue=None, *args)
    plot_list = [[y for y in x if y is not None] for x in plot_list]
    return bokeh.plotting.gridplot(plot_list, name=name)


def create_gradient(values, low, high, mid=None, val_range=None, intervals=100):
    
    # get anchor values
    vmin = val_range[0] if val_range is not None else values.min()
    vmax = val_range[1] if val_range is not None else values.max()
    vmid = (vmax+vmin)/2

    # get anchor colors
    cc = ColorConverter()
    low_color = np.array(cc.to_rgb(low))
    high_color = np.array(cc.to_rgb(high))
    mid_color = np.array(cc.to_rgb(mid)) if mid is not None else \
        (np.array(low_color)+np.array(high_color))/2
    
    # create array of values in order
    vals_low = np.linspace(vmin, vmid, intervals+1)
    vals_high = np.linspace(vmid, vmax, intervals+1)
    all_vals = np.unique(np.append(vals_low, vals_high))
    
    # calculate hex color representaitons for all values
    color_list = [low_color, mid_color, high_color]
    step_list = [(mid_color-low_color), (high_color-mid_color)]
    step_list = [l/intervals for l in step_list]
    color_range = []
    for v in all_vals:
        i = all_vals.tolist().index(v)
        j = 0 if i <= intervals else 1
        i = i - (intervals * j) - j
        nexti = color_list[0+j]+(step_list[j]*i)
        hsv = colorsys.rgb_to_hsv(*nexti)
        r, g, b = colorsys.hsv_to_rgb(*hsv)
        hex_value = '#%02X%02X%02X' % (r * 255, g * 255, b * 255)
        color_range.append(hex_value)

    # bring everything together
    output_values = [color_range[np.abs(v-all_vals).argmin()] for v in values]
    output_values = np.array(output_values)

    return output_values


def custom_components(plot_object, resources, element_id=None):
    """ Return HTML components to embed a Bokeh plot.

    The data for the plot is stored directly in the returned HTML.

    .. note:: The returned components assume that BokehJS resources
              are **already loaded**.

    Args:
        plot_object (PlotObject) : Bokeh object to render
            typically a Plot or PlotContext
        resources (Resources, optional) : BokehJS resources config

    Returns:
        (script, div)

    """
    ref = plot_object.get_ref()
    if element_id is None:
        elementid = str(uuid.uuid4())
    else:
        elementid = element_id
        plot_object._id = element_id

    js = bokeh.templates.PLOT_JS.render(
        elementid=elementid,
        modelid=ref["id"],
        modeltype=ref["type"],
        all_models=bokeh.protocol.serialize_json(plot_object.dump()),
    )
    script = bokeh.templates.PLOT_SCRIPT.render(
        plot_js=resources.js_wrapper(js),
    )
    div = bokeh.templates.PLOT_DIV.render(elementid=elementid)

    return bokeh.utils.encode_utf8(script), bokeh.utils.encode_utf8(div)


def custom_autoload_static(plot_object, resources, script_path, element_id=None):
    """ Return JavaScript code and a script tag that can be used to embed
    Bokeh Plots.

    The data for the plot is stored directly in the returned JavaScript code.

    Args:
        plot_object (PlotObject) :
        resources (Resources) :
        script_path (str) :

    Returns:
        (js, tag) :
            JavaScript code to be saved at ``script_path`` and a ``<script>``
            tag to load it

    Raises:
        ValueError

    """
    if resources.mode == 'inline':
        raise ValueError("autoload_static() requires non-inline resources")

    if resources.dev:
        raise ValueError("autoload_static() only works with non-dev resources")

    if element_id is None:
        elementid = str(uuid.uuid4())
    else:
        elementid = element_id
        plot_object._id = element_id

    js = bokeh.templates.AUTOLOAD.render(
        all_models=bokeh.protocol.serialize_json(plot_object.dump()),
        js_url=resources.js_files[0],
        css_files=resources.css_files,
        elementid=elementid,
    )

    tag = bokeh.templates.AUTOLOAD_STATIC.render(
        src_path=script_path,
        elementid=elementid,
        modelid=plot_object._id,
        modeltype=plot_object.__view_model__,
        loglevel = resources.log_level,
    )

    return bokeh.utils.encode_utf8(js), bokeh.utils.encode_utf8(tag)


def prep_parameters(keys=None, output_dir=None, id_prefix=None, js_folder=None, container=OrderedDict):
    if output_dir is None:
        use_dir = tempfile.tempdir
    else:
        use_dir = output_dir
        
    if id_prefix is None:
        out_file = tempfile.mkstemp(suffix='.html', dir=use_dir)[1]
    else:
        out_file = os.path.join(use_dir, id_prefix+'.html')
        
    if js_folder is not None:
        js_dir = os.path.join(use_dir, js_folder)
    else:
        js_dir = use_dir

    if container == dict:
        plot_keys = sorted(keys)
    elif container == OrderedDict:
        plot_keys = keys
    elif container == list:
        plot_keys = keys
    else:
        plot_keys = ['']

    return out_file, js_dir, plot_keys


def prep_plot(plot, key, id_prefix=None, js_folder='', js_dir=''):
    plot.canvas_height = plot.plot_height
    plot.canvas_width = plot.plot_width
    plot.outer_height = plot.plot_height
    plot.outer_width = plot.plot_width
    
    if id_prefix is not None:
        if type(key) == tuple:
            flag = '_'.join(key).rstrip('_')
            flag = re.sub('[^A-Za-z0-9-]+', '_', str(flag))
        else:
            flag = re.sub('[^A-Za-z0-9-]+', '_', str(key))
        plot._id = '_'.join([id_prefix, flag]).rstrip('_')

    r = bokeh.resources.CDN
    jspath = os.path.join(js_folder, '{id}.js'.format(id=plot._id))
    script, tag = custom_autoload_static(plot, r, jspath, plot._id)

    if not os.path.exists(js_dir):
        os.makedirs(js_dir)
    file_path = os.path.join(js_dir, plot._id+'.js')
    with open(file_path, 'w') as the_file:
        the_file.write(script)

    return plot


def prep_datatable(df, columns=None, js_dir='/', table_name='dataframe'):

    js_code = '''
    (function(global) {{
        window._datatables_onload_callbacks = [];

        datatablesjs_url = "http://cdn.datatables.net/1.10.2/js/jquery.dataTables.min.js"
        scripts = document.getElementsByTagName('script')
        for (i = 0; i < scripts.length; ++i) {{
            if (scripts[i].src == datatablesjs_url) {{
                scripts[i].parentNode.removeChild(scripts[i])
            }}
        }}

        function load_lib(url, callback){{
            window._datatables_onload_callbacks.push(callback);
            console.log("data tables.js not loaded, scheduling load and callback at", new Date());
            window._datatables_is_loading = true;
            var s = document.createElement('script');
            s.src = url;
            s.async = true;
            s.onreadystatechange = s.onload = function(){{
               window._datatables_onload_callbacks.forEach(function(callback){{callback()}});
            }};
            s.onerror = function(){{
                console.warn("failed to load library " + url);
            }};
            document.getElementsByTagName("head")[0].appendChild(s);
        }}

        var elt = document.getElementById("{table_id}");
        if(elt==null) {{
            console.log("ERROR: DataTable autoload script configured with elementid '{table_id}' but no matching script tag was found. ")
            return false;
        }}

        {variables}

        function inject_table() {{
            if (typeof $.fn.DataTable == "undefined") {{
                $.fn.DataTable = jQuery.fn.DataTable;
            }}
            if (typeof jQuery.fn.DataTable == "undefined") {{
                jQuery.fn.DataTable = $.fn.DataTable;
            }}
            var elem = $('.tableholder script#{table_id}').get(0);
            var table_elem = document.createElement('table');
            table_elem.setAttribute("class", "display compact")
            var header = table_elem.createTHead();
            var tr = document.createElement('TR');
            header.appendChild(tr);
            for (i = 0; i < table_columns.length; i++) {{
                var th = document.createElement('TH')
                th.appendChild(document.createTextNode(table_columns[i]['mData']));
                tr.appendChild(th);
            }}
            var par = elem.parentElement
            par.insertBefore(table_elem, elem.nextSibling)

            // Setup - add a text input to each footer cell
            $(table_elem).find('thead th').each(function () {{
                var title = $(table_elem).find('thead th').eq($(this).index()).text();
                $(this).html('<input type="text" placeholder= "' + title + '" />');
            }});

            $(table_elem).DataTable({{
                "bDestroy": true,
                "aaData": table_data,
                "aoColumns": table_columns,
                "iDisplayLength": 15,
                "aLengthMenu": [
                    [5, 15, 25, 50],
                    [5, 15, 25, 50]
                ]
            }});

            // Apply the search
            $(table_elem).DataTable().columns().eq(0).each(function (colIdx) {{
                $('input', $(table_elem).DataTable().column(colIdx).header()).on('keyup change', function () {{
                    $(table_elem).DataTable()
                        .column(colIdx)
                        .search(this.value)
                        .draw();
                }});
            }});
        }}

        function wait(name, callback) {{
                var interval = 10; // ms
                window.setTimeout(function() {{
                    if ($(name).length>0) {{
                        callback();
                    }} else {{
                        window.setTimeout(arguments.callee, interval);
                    }}
                }}, interval);
            }}


        load_lib(datatablesjs_url, function() {{
            console.log("DataTable autoload callback at", new Date())
            wait(".tableholder script#{table_id}", function(){{
                console.log("Injecting DataTable with id '{table_id}'")
                    inject_table()
            }});
        }});
    }}(this));
    '''

    name = 'table.js' if table_name is None else table_name+'_table.js'

    if type(columns) == OrderedDict:
        cols = columns.keys()
        cols_replace = columns.values()
    elif type(columns) == dict:
        cols = sorted(columns.keys())
        cols_replace = [columns[k] for k in cols]
    elif type(columns) == list:
        cols = columns
        cols_replace = None
    else:
        cols = None
        cols_replace = None

    if cols is not None:
        df = df[cols].drop_duplicates()
    if cols_replace is not None:
        df.columns = cols_replace

    js_table = 'var table_data = ' + df.to_json(orient='records') + ';\n'
    tabcols = ', '.join(["{{'mData': '{s}'}}".format(s=s) for s in df.columns])
    js_table += 'var table_columns = [' + tabcols + '];\n'

    js_code = js_code.format(table_id=(table_name+'_table'), variables=js_table)

    if not os.path.exists(js_dir):
        os.makedirs(js_dir)
    file_path = os.path.join(js_dir, name)
    with open(file_path, 'w') as the_file:
        the_file.write(js_code)


def make_tablemenu(table, columns=None, js_folder='/', js_dir='/', table_name='dataframe', key_columns=None, id_prefix=''):

    df = table.copy()

    js_code = '''
    (function(global) {{
        window._datatables_onload_callbacks = [];

        datatablesjs_url = "http://cdn.datatables.net/1.10.2/js/jquery.dataTables.min.js"
        scripts = document.getElementsByTagName('script')
        for (i = 0; i < scripts.length; ++i) {
            if (scripts[i].src == datatablesjs_url) {
                scripts[i].parentNode.removeChild(scripts[i])
            }
        }

        function load_lib(url, callback){{
            window._datatables_onload_callbacks.push(callback);
            console.log("data tables.js not loaded, scheduling load and callback at", new Date());
            window._datatables_is_loading = true;
            var s = document.createElement('script');
            s.src = url;
            s.async = true;
            s.onreadystatechange = s.onload = function(){{
               window._datatables_onload_callbacks.forEach(function(callback){{callback()}});
            }};
            s.onerror = function(){{
                console.warn("failed to load library " + url);
            }};
            document.getElementsByTagName("head")[0].appendChild(s);
        }}

        var elt = document.getElementById("{table_id}");
        if(elt==null) {{
            console.log("ERROR: DataTable autoload script configured with elementid '{table_id}' but no matching script tag was found. ")
            return false;
        }}

        {variables}

        function inject_table() {{
            if (typeof $.fn.DataTable == "undefined") {{
                $.fn.DataTable = jQuery.fn.DataTable;
            }}
            if (typeof jQuery.fn.DataTable == "undefined") {{
                jQuery.fn.DataTable = $.fn.DataTable;
            }}
            var elem = $('.menuholder script#{table_id}').get(0);
            var table_elem = document.createElement('table');
            table_elem.setAttribute("class", "display compact")
            var header = table_elem.createTHead();
            var tr = document.createElement('TR');
            header.appendChild(tr);
            for (i = 0; i < table_columns.length; i++) {{
                var th = document.createElement('TH')
                th.appendChild(document.createTextNode(table_columns[i]['mData']));
                tr.appendChild(th);
            }}
            var par = elem.parentElement
            par.insertBefore(table_elem, elem.nextSibling)

            // Setup - add a text input to each footer cell
            $(table_elem).find('thead th').each(function () {{
                var title = $(table_elem).find('thead th').eq($(this).index()).text();
                $(this).html('<input type="textarea" placeholder= "' + title + '" />');
            }});

            $(table_elem).DataTable({{
                "bDestroy": true,
                "aaData": table_data,
                "aoColumns": table_columns,
                "iDisplayLength": 15,
                "aLengthMenu": [
                    [5, 15, 25, 50],
                    [5, 15, 25, 50]
                ],
                "fnDrawCallback": function() {{
                     $("a[data-file]").off();
                     $("a[data-file]").on('click', function(){{
                         var filename = this.getAttribute('data-file');
                         var foldername = this.getAttribute('data-folder');
                         var targetname = this.getAttribute('data-target');
                         get_plot(filename, foldername, targetname)
                         wait(".bk-logo", function(){{
                            $(".bk-logo")[0].setAttribute("href", "http://www.infusiveintelligence.com/");
                            $(".bk-toolbar-button.help").remove();
                            var elements = document.getElementsByTagName('a');
                            for (var i = 0, len = elements.length; i < len; i++) {
                                elements[i].removeAttribute('title');
                            }
                        }});
                        var table = $("#"+targetname+" .tableholder")
                        if (table.length>0) {{
                            get_table(filename, foldername, targetname);
                        }}
                    }});
                }}
            }});

            // Apply the search
            $(table_elem).DataTable().columns().eq(0).each(function (colIdx) {{
                $('input', $(table_elem).DataTable().column(colIdx).header()).on('keyup change', function () {{
                    $(table_elem).DataTable()
                        .column(colIdx)
                        .search(this.value)
                        .draw();
                }});
            }});
        }}

        function wait(name, callback) {{
                var interval = 10; // ms
                window.setTimeout(function() {{
                    if ($(name).length>0) {{
                        callback();
                    }} else {{
                        window.setTimeout(arguments.callee, interval);
                    }}
                }}, interval);
            }}


        load_lib(datatablesjs_url, function() {{
            console.log("DataTable autoload callback at", new Date())
            wait(".menuholder script#{table_id}", function(){{
                console.log("Injecting DataTable with id '{table_id}'")
                    inject_table()
            }});
        }});
    }}(this));
    '''

    name = 'dataframe_menu.js' if table_name is None else table_name+'_menu.js'

    df['link'] = df[key_columns].apply(
        lambda x: (id_prefix+'_'+'_'.join(x)).strip('_'), axis=1).str.replace('[^A-Za-z0-9-]+', '_')
    columns['link'] = 'Plot'

    if type(columns) == OrderedDict:
        cols = columns.keys()
        cols_replace = columns.values()
    elif type(columns) == dict:
        cols = sorted(columns.keys())
        cols_replace = [columns[k] for k in cols]
    elif type(columns) == list:
        cols = columns
        cols_replace = None
    else:
        cols = None
        cols_replace = None

    if cols is not None:
        df = df[cols].drop_duplicates()
    if cols_replace is not None:
        df.columns = cols_replace

    js_table = 'var table_data = ' + df.to_json(orient='records') + ';\n'
    tabcols = ["{{'mData': '{s}'}}".format(s=s) for s in df.columns]
    tabcols[-1] = '''
    {{"mData": "Plot",
      "mRender": function(data, type, full) {{
        return '<a href="#" data-file="' + data + '" data-folder="{f}" data-target="{t}">show</a>';
        }}
    }}
    '''.format(f=js_folder.rstrip('/')+'/', t=id_prefix)
    tabcols = ', '.join(tabcols)
    js_table += 'var table_columns = [' + tabcols + '];\n'

    js_code = js_code.format(table_id=(table_name+'_menu'), variables=js_table)

    if not os.path.exists(js_dir):
        os.makedirs(js_dir)
    file_path = os.path.join(js_dir, name)
    with open(file_path, 'w') as the_file:
        the_file.write(js_code)


def make_slider(menu_items, id_prefix='', js_folder='', page_title=''):

    js_fold = (js_folder+'/') if js_folder != '' else js_folder

    menu = '''
    <nav>
      <ul class="list-unstyled main-menu">
        <li class="text-right"><a href="#" id="nav-close"></a></li>
        {menuitems}
      </ul>
    </nav>

    <div class="navbar navbar-inverse navbar-fixed-top">
        <a class="navbar-brand" href="#">{page_title}</a>
        <div class="navbar-header pull-right">
          <a id="nav-expander" class="nav-expander fixed">MENU</a>
        </div>
    </div>
    '''

    if all([isinstance(x, tuple) for x in menu_items]):
        menu_dict = DefaultOrderedDict(list)
        for f, s in menu_items:
            menu_dict[f].append(s)

        menuitems = []
        for key in menu_dict.keys():
            topitem = '''
            <li class="main-nav"><a href="#">{topname}</a>
                <ul class="list-unstyled">
                    {subitems}
                </ul>
            </li>
            '''

            subitems = []
            for val in menu_dict[key]:
                filename = '_'.join([id_prefix, '_'.join([str(key), str(val)])]).rstrip('_')
                filename = re.sub('[^A-Za-z0-9-]+', '_', str(filename))
                item = '''
                <li class="sub-nav hide">
                    <a href="#" data-file="{n}" data-folder="{f}" data-target="{t}">{k}</a>
                </li>
                '''
                item = item.format(n=filename, f=js_fold, k=val, t=id_prefix)
                subitems.append(item)

            topitem = topitem.format(topname=key, subitems='\n'.join(subitems))
            menuitems.append(topitem)

    elif all([isinstance(x, int) or isinstance(x, basestring) for x in menu_items]):
        menuitems = []
        for val in menu_items:
            filename = '_'.join([id_prefix, str(val)]).rstrip('_')
            filename = re.sub('[^A-Za-z0-9-]+', '_', str(filename))
            item = '''
            <li class="sub-nav">
                <a href="#" data-file="{n}" data-folder="{f}" data-target="{t}">{k}</a>
            </li>
            '''
            item = item.format(n=filename, f=js_fold, k=val, t=id_prefix)
            menuitems.append(item)
    else:
        print('Not yet implemented.')
        return ''

    menu = menu.format(menuitems='\n'.join(menuitems), page_title=page_title)
    return menu


def prep_body(plot_keys, id_prefix=None, js_folder='', footer=None, table=True, page_title='', menu=None, js_dir='/'):

    html_body = '''
    <body>
        {navigation}
        <div class="container-fluid" id="{id}">
            <div class="row">
                {content}
            </div>
        </div>
        {footer}
    </body>
    '''
    content = ''
    if menu is not None:
        make_tablemenu(
            menu['data'], columns=menu['columns'], key_columns=menu['keys'], js_folder=js_folder, js_dir=js_dir,
            id_prefix=id_prefix)
        navigation = ''
        content += '''
        <div class="menuholder col-md-4">
            <script src="{s}" id="dataframe_menu"></script>
        </div>
        '''.format(s=os.path.join(js_folder, 'dataframe_menu.js'))
        content += '<div class="plotholder col-md-8">\n</div>'

    else:
        navigation = make_slider(
            plot_keys, id_prefix=id_prefix, js_folder=js_folder, page_title=page_title)

        content = '<div class="plotholder col-md-6">\n</div>'
        if table:
            content += '\n'
            content += '<div class="tableholder col-md-6"></div>'

    footer = '' if footer is None else '<div id="footer"><hr><p>{f}</p></div>\n'.format(f=footer)

    html_body = html_body.format(navigation=navigation, id=id_prefix, content=content, footer=footer)

    return html_body


def arrange_plots(
    plots, browser=None, new="tab", id_prefix=None, output_dir=None, js_folder='js', show=True, table_cols=None,
        footer=None, page_title=None, menu=None):

    plots_type = type(plots)
    keys = plots.keys() if hasattr(plots, 'keys') else range(len(plots))

    out_file, js_dir, plot_keys = prep_parameters(
        keys=keys, output_dir=output_dir, id_prefix=id_prefix, js_folder=js_folder, container=plots_type)

    for key in plot_keys:
        plot = plots[key] if hasattr(plots, '__iter__') else plots
        plot = prep_plot(plot, key, id_prefix=id_prefix, js_dir=js_dir, js_folder=js_folder)
        do_tables = table_cols is not None
        if do_tables:
            dfs = [ref for ref in plot.references() if type(ref) == bokeh.objects.ColumnDataSource]
            ind = np.argmax([set(df.data.keys()).intersection(table_cols.keys()).__len__() for df in dfs])
            df = pd.DataFrame(dfs[ind].data)
            prep_datatable(df, columns=table_cols, js_dir=js_dir, table_name=plot._id)
    if show | (menu is not None):
        html_head = HTML_HEAD + HTML_SCRIPTS + HTML_CSS + '\n</head>\n'

        html_body = prep_body(
            plot_keys, id_prefix=id_prefix, js_folder=js_folder, table=do_tables, footer=footer,
            page_title=page_title, menu=menu, js_dir=js_dir)
    if show:
        with open(out_file, 'w') as the_file:
            the_file.write(html_head)
            the_file.write(html_body)
            the_file.write('\n</html>')

        controller = bokeh.browserlib.get_browser_controller(browser=browser)
        new_param = {'tab': 2, 'window': 1}[new]
        controller.open("file://" + out_file, new=new_param)