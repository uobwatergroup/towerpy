# Configuration file for the Sphinx documentation builder.

import os
import sys

sys.path.insert(0, os.path.abspath('../'))

# -- Project information

project = 'Towerpy'
author = 'DSR-MARR-TowerpyCom'

release = '2022'
version = '1.0.5'

# -- General configuration

extensions = [
    'sphinx.ext.doctest',
    'sphinx.ext.autodoc',
    'sphinx.ext.autosummary',
    'sphinx.ext.intersphinx',
    'sphinx.ext.mathjax',
    'sphinx.ext.viewcode',
    'nbsphinx',
    'myst_nb',
    'autoapi.extension',
    'sphinx_copybutton',
]

# Document Python Code
autoapi_type = 'python'
autoapi_dirs = ['../../towerpy']

intersphinx_mapping = {
    'python': ('https://docs.python.org/3/', None),
    'sphinx': ('https://www.sphinx-doc.org/en/master/', None),
    'numpy': ("https://numpy.org/doc/stable/", None),
    'scipy': ("https://docs.scipy.org/doc/scipy/reference/", None),
    'matplotlib': ("https://matplotlib.org/stable/", None),
    'cartopy': ("https://scitools.org.uk/cartopy/docs/latest/", None)
    }

intersphinx_disabled_domains = ['std']

templates_path = ['_templates']

exclude_patterns = ['_build', '**.ipynb_checkpoints']

# -- Options for HTML output
html_theme = 'pydata_sphinx_theme'
html_theme_options = {
    'github_url': 'https://github.com/uobwatergroup/towerpy'
    }
html_title = project
html_show_sourcelink = False
html_logo = '_static/towerpy_logosd.png'

# Main document.
master_doc = 'index'

# Suffix of source filenames.
source_suffix = ['.rst', '.md', '.ipynb']

# Generate the API documentation when building
autoclass_content = 'both'

autosummary_generate = True
autosummary_imported_members = True

# -- nbsphinx specifics --
# do not execute notebooks ever while building docs
nbsphinx_execute = 'never'


# -- Options for EPUB output
epub_show_urls = 'footnote'


