# Configuration file for the Sphinx documentation builder.

import os
import sys
sys.path.insert(0, os.path.abspath('.'))

# -- Project information

project = 'Towerpy'
author = 'Towerpy'

release = '2022'
version = '1.0.0'

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
]

intersphinx_mapping = {
    'python': ('https://docs.python.org/3/', None),
    'sphinx': ('https://www.sphinx-doc.org/en/master/', None),
}
intersphinx_disabled_domains = ['std']

templates_path = ['_templates']

exclude_patterns = ["_build", "**.ipynb_checkpoints"]

# -- Options for HTML output
plot_html_show_source_link = False
html_theme = 'pydata_sphinx_theme'
html_theme_options = {
    "github_url": "https://github.com/uobwatergroup/towerpy",
}

master_doc = "index"
source_suffix = [".rst", ".md", ".ipynb"]


# Generate the API documentation when building
autoclass_content = "both"

autosummary_generate = True
autosummary_imported_members = True

# -- Options for EPUB output
epub_show_urls = 'footnote'


