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

DSTYPE = bokeh.objects.ColumnDataSource

HTML_HEAD = '''
<!DOCTYPE html>
<html>
<head>

<link rel="stylesheet" href="http://maxcdn.bootstrapcdn.com/bootstrap/3.2.0/css/bootstrap.min.css">
<link rel="stylesheet" href="http://maxcdn.bootstrapcdn.com/bootstrap/3.2.0/css/bootstrap-theme.min.css">
<link rel="stylesheet" href="http://cdn.pydata.org/bokeh-0.5.1.css" type="text/css" />
<link rel="stylesheet" href="http://cdn.datatables.net/1.10.2/css/jquery.dataTables.min.css">

<script src="http://ajax.googleapis.com/ajax/libs/jquery/2.1.1/jquery.min.js"></script>
<script src="http://maxcdn.bootstrapcdn.com/bootstrap/3.2.0/js/bootstrap.min.js"></script>
<script type="text/javascript" src="http://cdn.pydata.org/bokeh-0.5.1.js"></script>
<script type="text/javascript" src="http://cdn.datatables.net/1.10.2/js/jquery.dataTables.min.js"></script>
'''

HTML_SCRIPTS = '''
<script type="text/javascript">

    function get_plot(plotid, folder, target, callback) {
        var s = document.createElement("script");
        s.type = "text/javascript";
        s.src = folder.concat(plotid).concat(".js");
        s.id = plotid;
        s.async = "true";
        s.setAttribute("data-bokeh-data", "static");
        s.setAttribute("data-bokeh-modelid", plotid);
        s.setAttribute("data-bokeh-modeltype", "Plot");
        s.onreadystatechange = s.onload = function () {
            var state = s.readyState;
            if (!callback.done && (!state || /loaded|complete/.test(state))) {
                callback.done = true;
                console.log('Done waiting')
                callback();
            }
        };
        var elem = $('#'+target).find('.plotholder')[0]
        if (elem.children.length == 0) {
            elem.appendChild(s)
        } else {
            var old_child = elem.children[0]
            elem.replaceChild(s, old_child);
        }
    }

    get_table = function (tableid, folder, target, table_columns) {
            if (typeof all_models!="undefined") {
                delete all_models
            }
            var oldkeys = keys(table_columns)
            var newkeys = values(table_columns)
            filepath = folder.concat(tableid).concat(".js");
            var rawFile = new XMLHttpRequest();
            rawFile.open("GET", filepath, false);
            rawFile.onreadystatechange = function () {
                if(rawFile.readyState === 4) {
                    if(rawFile.status === 200 || rawFile.status == 0) {
                        var match = rawFile.responseText;
                        var match = match.match(/all_models.*];/)[0]
                        eval(match)
                    }
                }
            }
            rawFile.send(null);

            all_models = $.grep(all_models, function(e){ return e.type == "ColumnDataSource"; });

            var obj_lengths = [];
            for (var i = 0; i < all_models.length; i++) {
                var nmatch = oldkeys.filter(function(n) {
                    return all_models[i].attributes['column_names'].indexOf(n) != -1
                }).length;
                obj_lengths.push(nmatch);
            }

            var i = obj_lengths.indexOf(Math.max.apply(Math, obj_lengths));
            var all_models= all_models[i].attributes['data']
            var table_data = []
            for(var i in oldkeys) {
                kk = oldkeys[i]
                table_data.push(all_models[kk]);
              }
            }

            var elem = $('#'+target + ' .dataframe')[0]
            if (jQuery.fn.dataTable.fnIsDataTable(elem)) {
                jQuery(elem).dataTable().fnClearTable();
                jQuery(elem).dataTable().fnAddData(table_data);
            } else {
                var header = elem.createTHead();
                var tr = document.createElement('TR');
                header.appendChild(tr);
                for (i = 0; i < table_columns.length; i++) {
                    var th = document.createElement('TH')
                    th.appendChild(document.createTextNode(table_columns[i]['mData']));
                    tr.appendChild(th);
                }

                // Setup - add a text input to each footer cell
                $(elem).find('thead th').each(function () {
                    var title = $(elem).find('thead th').eq($(this).index()).text();
                    $(this).html('<input type="text" placeholder= "' + title + '" />');
                });

                jQuery(elem).DataTable({
                    "bDestroy": true,
                    "aaData": table_data,
                    "aoColumns": newkeys,
                    "iDisplayLength": 15,
                    "aLengthMenu": [
                        [5, 15, 25, 50],
                        [5, 15, 25, 50]
                    ]
                });
            }

            // Apply the search
            jQuery(elem).DataTable().columns().eq(0).each(function (colIdx) {
                $('input', jQuery(elem).DataTable().column(colIdx).header()).on('keyup change', function () {
                    jQuery(elem).DataTable()
                            .column(colIdx)
                            .search(this.value)
                            .draw();
                });
            });

        }

    function load_menu(menu_items) {
        menu = $('#menu')
        menu.empty();

        $.each(menu_items, function(){
            console.log(this.subitems)
            if (this.subitems==null) {
                bullet = $('<li />')
                $("<a />")
                .attr("data-file", this.target+'_'+this.suffix)
                .attr("data-folder", this.folder)
                .attr("data-target", this.target)
                .attr("href", "#")
                .attr("class", this.class)
                .html(this.name)
                .appendTo(bullet);
                bullet.appendTo(menu);
            } else {
                bullet = $('<li />')
                $("<a />")
                .attr("menu-pointer", this.suffix)
                .attr("href", "#")
                .html(this.name)
                .appendTo(bullet);
                bullet.appendTo(menu);
            }
        });
    }

    function load_submenu() {
        var pointer = this.getAttribute('menu-pointer')
        var obj  = $.grep(menu_items, function(e){ return e.suffix ==pointer; })[0];
        if (obj.subitems!=null) {
            var thissuffix = obj.suffix
            var thisfolder = obj.folder
            var thistarget = obj.target
            var thisclass = obj.class
            menu.empty();
            header = $("<a />")
            .click(function(){load_menu(menu_items)})
            .attr("class", "backup")
            .html('Go Back')
            .appendTo(bullet);
            header.appendTo(menu);
            $('<hr>').appendTo(menu);
            $.each(obj.subitems, function(){
                 bullet = $('<li />')
                $("<a />")
                .attr("data-file", thistarget+'_'+this.suffix)
                .attr("data-folder", thisfolder)
                .attr("data-target", thistarget)
                .attr("href", "#")
                .attr("class", thisclass)
                .html(this.name)
                .appendTo(bullet);
                bullet.appendTo(menu);
            });
        }
    }

    window.onload = function () {
        var plotanchors = document.getElementsByClassName('clicker plot');

        var items = {};
        $(plotanchors).each(function() {
            items[$(this).attr('data-target')] = true;
        });

        for(var target in items) {
            var plotanchors = $('.clicker.plot[data-target="'+target+'"]');
            var tableanchors = $('.clicker.table[data-target="'+target+'"]');
            var firstname = plotanchors[0].getAttribute('data-file');
            var firstfolder = plotanchors[0].getAttribute('data-folder');
            var firsttarget = plotanchors[0].getAttribute('data-target');

            if (tableanchors.length > 0) {
                var click_handler = function () {
                    var filename = this.getAttribute('data-file');
                    var foldername = this.getAttribute('data-folder');
                    var targetname = this.getAttribute('data-target');
                    get_plot(filename, foldername, targetname, function () {
                        get_table(table_data, table_columns, targetname, table_columns);
                        $("#"+targetname+" .bokeh.plotview").width(Math.round($(".bokeh.plotview").width() * 0.9));
                        var window_width = $(window).width()
                        var plot_width = $("#"+targetname+" .bokeh.plotview").width()
                        $("#"+targetname+" .tableholder").width((window_width-plot_width)*0.9);
                        $(".bk-logo")[0].setAttribute("href", "http://www.infusiveintelligence.com/")
                    });
                }
            } else {
                var click_handler = function () {
                    var filename = this.getAttribute('data-file');
                    var foldername = this.getAttribute('data-folder');
                    var targetname = this.getAttribute('data-target');
                    get_plot(filename, foldername, targetname, function () {
                        console.log('No table')
                    });
                }
            }

            for (var i = 0; i < plotanchors.length; i++) {
                plotanchors[i].onclick = click_handler
            }

            plotanchors[0].click()
        }
    }

    $(function() {
      // Toggle main on click
      $('.menu-toggle a').click(function() {
          $('#entrance-nav').toggleClass("hider");
          $('.page-canvas').toggleClass('show-main-nav');
          return false;
      });
    });

    $(function() {
      // Toggle sub on click
      $('.submenu-toggle a').click(function() {
          $('.page-canvas').removeClass("show-main-nav");
          $('.page-canvas').addClass("show-sub-nav");
          $(".page-canvas>div.page-submenu").removeClass("show-sub-nav");
          var id = $(this).attr("href");
          $('.page-submenu '+id).addClass('show-sub-nav');
          return false;
      });
    });

    $(function() {
      // Toggle sub on click
      $('.menu-reset a').click(function() {
          $(".page-canvas>div.page-submenu").removeClass("show-sub-nav");
          $('.page-canvas').removeClass("show-sub-nav");
          $('.page-canvas').addClass("show-main-nav");
          return false;
      });
    });

    $(function() {
      // Toggle sub on click
      $('.menu-submit a').click(function() {
          $(".page-canvas>div.page-submenu").removeClass("show-sub-nav");
          $('.page-canvas').removeClass("show-sub-nav");
          $('#entrance-nav').toggleClass("hider");
          return false;
      });
    });
}

$(window).resize(function(){
    var window_width = $(window).width()
    var plot_width = $(".bokeh.plotview").width()
    var pad_left = parseInt($('.container').css('padding-left').replace(/px/,""))
    var pad_right = parseInt($('.container').css('padding-right').replace(/px/,""))
    $("#tableholder").width(window_width-pad_left-pad_right-plot_width);
    });

</script>
'''

HTML_CSS = '''
<style type="text/css">
body {
     font-family: "Century Gothic", CenturyGothic, AppleGothic, sans-serif;
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

.page-canvas {
  width: 100%;
  height: 100%;
  position: relative;
  -webkit-transform: translateX(0);
  transform: translateX(0);
  -webkit-transform: translate3d(0, 0, 0);
  transform: translate3d(0, 0, 0);
  -webkit-backface-visibility: hidden;
  backface-visibility: hidden;
  float: left;
}
.show-main-nav .page-canvas, .show-sub-nav .page-canvas {
  -webkit-transform: translateX(300px);
  transform: translateX(300px);
  transform: translate3d(300px, 0, 0);
  -webkit-transform: translate3d(300px, 0, 0);
}

.page-menu, .page-submenu {
  width: 275px;
  height:100vh;
  position: absolute;
  top: 0;
  left: -300px;
  background: white;
  padding: 0px;
  box-shadow: 2px 0px 2px black;
  overflow: hidden;
  z-index: 10000;
  float: left;
}

.menu-options {
  width: 275px;
  height: 90%;
  position: relative;
  top: -15px;
  background: white;
  overflow: scroll;
}

.slider .page-menu, .slider .page-submenu {
  -webkit-transform: translateX(0);
  transform: translateX(0);
  -webkit-transform: translate3d(0, 0, 0);
  transform: translate3d(0, 0, 0);
}

.slider .show-main-nav .page-menu, .slider .show-sub-nav .page-submenu {
  -webkit-transition: 200ms ease all;
  transition: 200ms ease all;
  -webkit-transform: translateX(300px);
  transform: translateX(300px);
  -webkit-transform: translate3d(300px, 0, 0);
  transform: translate3d(300px, 0, 0);
}
.slider .show-main-nav .page-canvas, .slider .show-sub-nav .page-canvas {
  -webkit-transform: translateX(0);
  transform: translateX(0);
  -webkit-transform: translate3d(0, 0, 0);
  transform: translate3d(0, 0, 0);
}

.menu-toggle a, .menu-reset a  {
    font-size: 40px;
    text-decoration: none;
    color: #FF4738;
    display: inline-block;
    padding: 0px;
    text-indent: 0px;
    margin-top: 0px;
    margin-bottom: 0px;
    line-height: 0px;
}

.main-nav a{
  text-indent: 100px;
}

.menu-toggle.hider a {
    color: transparent;
    pointer-events: none;
    cursor: default;
}
.menu-options>ul a {
  font-size: 14px;
  text-decoration: none;
  color: #808080;
  display: inline-block;
}

menu-options>ul  a:hover {
  font-size: 16px;
  text-decoration: none;
  color: #FF4738;
}

ul {
    list-style-type: none;
    padding: 0px 0px 0px 30px;
}

.container {
    padding-left: 10px;
    padding-right: 20px;
    margin: 0px;
    width: 100%;
}

#plotholder {
  float:left;
}

#tableholder {
  float:left;
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

.dataTables_wrapper .dataTables_length, .dataTables_wrapper .dataTables_filter, .dataTables_wrapper .dataTables_info, .dataTables_wrapper .dataTables_processing, .dataTables_wrapper .dataTables_paginate {
  font-size: 11px;
}

.bk-toolbar-button:focus,
.bk-toolbar-button:active:focus,
.bk-toolbar-button.bk-bs-active:focus {
  outline: thin dotted;
  outline: 0px auto -webkit-focus-ring-color;
  outline-offset: 0px;
}

.bk-logo-medium {
  width: 35px;
  height: 35px;
  background-image: url(data:image/png;base64,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);
}
.bk-sidebar {
  float: left;
  left: -20px;
  margin-top: 50px;
}

.bk-button-bar > .bk-toolbar-button {
  width: 40px;
  height: 40px;
}

.bk-toolbar-button > .bk-btn-icon {
  width: 28px;
  height: 28px;
  float: left;
  left: -25%;
  transform: translateY(0%);
}

.bk-toolbar-button > span.tip {
  transition: all 0.3s ease;
  -webkit-transition: all 0.3s ease;
  -moz-transition: all 0.3s ease;
  -o-transition: all 0.3s ease;
  font-family: "Century Gothic", CenturyGothic, AppleGothic, sans-serif;
}

.bk-toolbar-button:hover > span.tip {
  left: 40px;
  font-size: 100%;
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

.plottitle {
  padding-left: 0px;
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

    def __missing__ (self, key):
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
    return vals[order], vecs[:,order]

def calculate_ovals(x, y, sd=3):

    cov = np.cov(x, y)
    lambda_, v = eigsorted(cov)
    lambda_ = np.sqrt(lambda_)

    x = np.mean(x)
    y = np.mean(y)
    width = lambda_[0]*sd*2
    height = lambda_[1]*sd*2
    angle = np.degrees(np.arctan2(*v[:,0][::-1]))

    out = pd.Series({
        'x':x, 'y':y,
        'width':width, 'height':height,
        'angle':angle}).astype(float)
    return out

def change_axes(**kwargs):

    p = bokeh.plotting.curplot()
    if p is None:
        return None

    try:
        axis_obj = bokeh.objects.Axis
        x_axis = [obj for obj in p.above + p.below if isinstance(obj, axis_obj)]
        y_axis = [obj for obj in p.left + p.right if isinstance(obj, axis_obj)]
    except:
        axes = bokeh.plotting.axis()
        x_axis = [axes[0]]
        y_axis = [axes[1]]

    xtags = sum([1 if k.startswith('x_') else 0 for k in kwargs.keys()])
    ytags = sum([1 if k.startswith('y_') else 0 for k in kwargs.keys()])

    if xtags>0:
        xs = [(k[2:],v) for k,v in kwargs.items() if k.startswith('x_')]
        for k,v in xs:
            x_axis[0].__setattr__(k, v)
    if ytags>0:
        ys = [(k[2:],v) for k,v in kwargs.items() if k.startswith('y_')]
        for k,v in ys:
            y_axis[0].__setattr__(k, v)

    if (xtags+ytags)<len(kwargs):
        boths = xs = [(k,v) for k,v in kwargs.items() if
            (k.startswith('x_')+k.startswith('y_'))==0]
        for k,v in boths:
            x_axis[0].__setattr__(k, v)
            y_axis[0].__setattr__(k, v)

    bind = x_axis+y_axis
    return bind

def change_legend(**kwargs):
    legend = bokeh.plotting.legend()[0]
    for k,v in kwargs.items():
        legend.__setattr__(k, v)
    return legend

def set_hover(values):
    hover = [t for t in bokeh.plotting.curplot().tools if
        isinstance(t, bokeh.objects.HoverTool)][0]
    hover.tooltips = OrderedDict(values)
    return hover

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
        j = 0 if i<=(intervals) else 1
        i = i-(intervals*j)-j
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
    ''' Return HTML components to embed a Bokeh plot.

    The data for the plot is stored directly in the returned HTML.

    .. note:: The returned components assume that BokehJS resources
              are **already loaded**.

    Args:
        plot_object (PlotObject) : Bokeh object to render
            typically a Plot or PlotContext
        resources (Resources, optional) : BokehJS resources config

    Returns:
        (script, div)

    '''
    ref = plot_object.get_ref()
    if element_id is None:
        elementid = str(uuid.uuid4())
    else:
        elementid = element_id
        plot_object._id = element_id

    js = bokeh.templates.PLOT_JS.render(
        elementid = elementid,
        modelid = ref["id"],
        modeltype = ref["type"],
        all_models = bokeh.protocol.serialize_json(plot_object.dump()),
    )
    script = bokeh.templates.PLOT_SCRIPT.render(
        plot_js = resources.js_wrapper(js),
    )
    div = bokeh.templates.PLOT_DIV.render(elementid=elementid)

    return bokeh.utils.encode_utf8(script), bokeh.utils.encode_utf8(div)

def custom_autoload_static(plot_object, resources, script_path, element_id=None):
    ''' Return JavaScript code and a script tag that can be used to embed
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

    '''
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
        all_models = bokeh.protocol.serialize_json(plot_object.dump()),
        js_url = resources.js_files[0],
        css_files = resources.css_files,
        elementid = elementid,
    )

    tag = bokeh.templates.AUTOLOAD_STATIC.render(
        src_path = script_path,
        elementid = elementid,
        modelid = plot_object._id,
        modeltype = plot_object.__view_model__,
    )

    return bokeh.utils.encode_utf8(js), bokeh.utils.encode_utf8(tag)


def prep_parameters(keys=None, output_dir=None, output_file=None,
    js_folder=None, container=OrderedDict):
    if output_dir is None:
        use_dir = tempfile.tempdir
    else:
        use_dir = output_dir

    if output_file is None:
        out_file = tempfile.mkstemp(suffix='.html', dir=use_dir)[1]
    else:
        out_file = os.path.join(use_dir, output_file)

    if js_folder is not None:
        js_dir = os.path.join(use_dir, js_folder)
    else:
        js_folder = ''
        js_dir = use_dir

    if container==dict:
        plot_keys = sorted(keys)
    elif container==OrderedDict:
        plot_keys = keys
    elif container==list:
        plot_keys = keys
    else:
        plot_keys = ['']

    return out_file, js_dir, plot_keys

def prep_plot(plot, key, id_prefix=None, ):
    plot.canvas_height = plot.plot_height
    plot.canvas_width = plot.plot_width
    plot.outer_height = plot.plot_height
    plot.outer_width = plot.plot_width

    if id_prefix is not None:
        if type(key)==tuple:
            flag = '_'.join(key).rstrip('_')
            flag = re.sub('[^A-Za-z0-9-]+', '_', str(flag))
        else:
            flag = re.sub('[^A-Za-z0-9-]+', '_', str(key))
        plot._id = '_'.join([id_prefix, flag]).rstrip('_')

    return plot

def prep_script(plot, mode='cdn', js_folder='', js_dir=''):
    r = bokeh.resources.INLINE if mode=='inline' else bokeh.resources.CDN
    if mode=='inline':
        script, tag = custom_components(plot, r, plot._id)
    elif mode=='cdn':
        jspath = os.path.join(js_folder, '{id}.js'.format(id=plot._id))
        script, tag = custom_autoload_static(plot, r, jspath, plot._id)

    if (mode=='cdn'):
        if not os.path.exists(js_dir):
            os.makedirs(js_dir)
        file_path = os.path.join(js_dir, plot._id+'.js')
        with open(file_path, 'w') as the_file:
            the_file.write(script)

    return script, tag

def make_setup(menu_items, id_prefix='', table=False, js_folder=''):

    js_fold = js_folder.rstrip('/')+'/' if js_folder!='' else js_folder
    classnames = 'clicker plot table' if table else 'clicker plot'

    menu_dict = DefaultOrderedDict(list)
    if all([(len(x)==2) and (type(x)!=str) for x in menu_items]):
        for f, s in menu_items:
            news = {'name':str(s), 'suffix':re.sub('[^A-Za-z0-9-]+', '_', str(s))}
            menu_dict[f].append(news)
    elif all([(len(x)==1) or (type(x)==str) for x in menu_items]):
        for f in menu_items:
            menu_dict[f] = None
    else:
        print 'Not yet implemented.'
        return ''

    items = []
    for f in menu_dict.keys():
        it = {}
        it['name'] = str(f)
        it['suffix'] = re.sub('[^A-Za-z0-9-]+', '_', str(f))
        it['folder'] = js_fold
        it['target'] = id_prefix
        it['subitems'] = menu_dict[f]
        it['class'] = classnames

        items.append(it)

    items = 'menu_items = ' + json.dumps(items) + ';'

    return items


def prep_body(plot_keys, id_prefix=None, js_folder='', footer=None, table=True):

    html_body = '''
    <body>
        <div class='container-fluid'>
            {content}
        </div>
        {footer}
    </body>
    '''

    content = '''
    <div class="navigation">
        <a href="#"><span class="glyphicon glyphicon-chevron-right"></a>
    </div>
    <div class="page-menu">
        <ul id="menu"></ul>
    </div>
    <div class="plotholder">\n</div>
    '''

    if table:
        content += '''
        <div class="tableholder">
            <table class="display compact dataframe"></table>
        </div>
        '''

    if footer is not None:
        footer = '<div id="footer"><hr><p>{f}</p></div>\n'.format(f=footer)
    else:
        footer = ''

    html_body = html_body.format(id=id_prefix, content=content, footer=footer)

    return html_body

def arrange_plots(plots, browser=None, new="tab", mode='cdn', id_prefix=None,
    output_dir=None, output_file=None, js_folder='js',
    show=True, table_cols=None, footer=None, page_title=None):

    plots_type = type(plots)

    try:
        keys = plots.keys()
    except:
        keys = range(len(plots))


    out_file, js_dir, plot_keys = prep_parameters(
        keys=keys, output_dir=output_dir, output_file=output_file,
        js_folder=js_folder, container=plots_type)

    scripts = []
    tags = []
    for k in plot_keys:
        try:
            plot = plots[k]
        except:
            plot = plots
        plot = prep_plot(plot, k, id_prefix=id_prefix)
        script, tag = prep_script(plot, mode=mode, js_dir=js_dir,
            js_folder=js_folder)
        scripts.append(script)
        tags.append(tag)

    do_tables = table_cols is not None
    if do_tables:
        setup = 'table_columns = ' + json.dumps(table_cols) + ';\n'
        setup += make_setup(plot_keys, id_prefix=id_prefix, js_folder=js_folder,
            table=do_tables)
        if (mode=='cdn'):
            if not os.path.exists(js_dir):
                os.makedirs(js_dir)
            file_path = os.path.join(js_dir, 'setup.js')
            with open(file_path, 'w') as the_file:
                the_file.write(setup)

    setup_script = '<script type="text/javascript" src="{src}"></script>'.format(
        src=os.path.join(js_folder, 'setup.js'))

    if show:
        html_head = HTML_HEAD + setup_script + HTML_SCRIPTS + HTML_CSS

        if mode=='inline':
            html_head = '\n'.join([html_head]+scripts)
        html_head += '\n</head>\n'

        html_body = prep_body(plot_keys, id_prefix=id_prefix, js_folder=js_folder,
            table=do_tables, footer=footer)

        with open(out_file, 'w') as the_file:
            the_file.write(html_head)
            the_file.write(html_body)
            the_file.write('\n</html>')


        controller = bokeh.browserlib.get_browser_controller(browser=browser)
        new_param = {'tab': 2, 'window': 1}[new]
        controller.open("file://" + out_file, new=new_param)