"""
The Dependency Abstraction Layer (Dependency Firewall).

This module serves as the sole authorized gateway for all external dependencies 
and Python standard library imports across the entire AIDflow ecosystem. 
Following a strict architectural mandate, no module within the AIDflow package 
is permitted to import third-party libraries or native Python modules directly; 
instead, they must consume these dependencies via the `.deps` abstraction layer.
"""

CSS_IT = 'AIG-datatable'

from itables import ( 
    show, 
    to_html_datatable,
    JavascriptFunction
)

import itables.options as opt
from functools import wraps


opt.showIndex  = True
opt.maxColumns = 500  
opt.maxBytes   = '256KB'
opt.warn_on_undocumented_option = False


def _itables_css(kwargs:dict, update_css:str) -> str:
    base_classes = kwargs.pop('classes', CSS_IT)
    classes = f"display {base_classes}" 

    if update_css:
        cls_ext   = 'table-custom-extras-css'
        classes  += f" {cls_ext}"
        css_ext   = f".dt-container table.dataTable.{cls_ext} {update_css.strip()}"
        css_exist = kwargs.get('css', '')
        kwargs['css'] = f"{css_exist}\n{css_ext}".strip()

    return classes
    
@wraps(show)
def itables_show(*args, update_css:str=None, **kwargs) -> None:
    classes = _itables_css(kwargs, update_css)
    return show(*args, **kwargs, allow_html=True, classes=classes)

@wraps(to_html_datatable)
def itables_html(*args, update_css:str=None, **kwargs) -> str:
    classes = _itables_css(kwargs, update_css)
    return to_html_datatable(*args, **kwargs, allow_html=True, classes=classes)