import os
import sys


sys.path.insert(0, os.path.abspath('..'))


project = 'ttopt'
copyright = '2021-2022'
author = 'Andrei Chertkov'
language = 'en'
html_theme = 'alabaster'
html_favicon = '_static/favicon.ico'
html_theme_options = {
    'logo': 'favicon.ico',
    'logo_name': True,
    'show_powered_by': False,
    'show_relbars': False,
}
extensions = [
    'sphinx.ext.imgmath',
    'sphinx.ext.graphviz',
    'sphinx.ext.autodoc',
    'sphinx.ext.napoleon',
    'sphinx.ext.todo',
    'sphinx.ext.inheritance_diagram',
    'sphinx.ext.viewcode',
    'm2r2'
]
templates_path = [
    '_templates',
]
exclude_patterns = [
    '_build',
    'Thumbs.db',
    '.DS_Store',
]
html_static_path = [
    '_static',
]

napoleon_google_docstring = True
napoleon_numpy_docstring = False
napoleon_include_init_with_doc = True
napoleon_include_private_with_doc = True
napoleon_include_special_with_doc = True
napoleon_use_admonition_for_examples = True
napoleon_use_admonition_for_notes = True
napoleon_use_admonition_for_references = True
napoleon_use_ivar = True
napoleon_use_param = True
napoleon_use_rtype = True
napoleon_type_aliases = None

todo_include_todos = True
