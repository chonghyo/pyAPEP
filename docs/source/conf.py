# Configuration file for the Sphinx documentation builder.
#
# This file only contains a selection of the most common options. For a full
# list see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Path setup --------------------------------------------------------------

# If extensions (or modules to document with autodoc) are in another directory,
# add these directories to sys.path here. If the directory is relative to the
# documentation root, use os.path.abspath to make it absolute, like shown here.
#

from __future__ import absolute_import
import sys, os
# sys.path.insert(0, os.path.abspath('.'))   # Default path
sys.path.insert(0, os.path.abspath('../..'))   # 필자의 path
import sphinx_rtd_theme

# -- Project information -----------------------------------------------------

project = 'pyAPEP'
copyright = '2022, PSID'
author = 'PSID'


# -- General configuration ---------------------------------------------------

# Add any Sphinx extension module names here, as strings. They can be
# extensions coming with Sphinx (named 'sphinx.ext.*') or your custom
# ones.
extensions = [
    'sphinx.ext.autosummary',
    'sphinx.ext.doctest',
    'sphinx.ext.intersphinx',
    'sphinx.ext.todo',
    'sphinx.ext.coverage',
    'sphinx.ext.mathjax',
    'sphinx.ext.ifconfig',
    'sphinx.ext.napoleon',
    "sphinx_rtd_theme",
    'sphinx.ext.autodoc',
    'sphinx.ext.imgmath',
    'sphinx.ext.ifconfig',
    'sphinx.ext.viewcode',
    'sphinx.ext.githubpages',
    'recommonmark',
]

# Add any paths that contain templates here, relative to this directory.
templates_path = ['_templates']

# List of patterns, relative to source directory, that match files and
# directories to ignore when looking for source files.
# This pattern also affects html_static_path and html_extra_path.
source_suffix = ['.rst', '.md']
master_doc = 'index'
language = 'en'
exclude_patterns = []
pygments_style = 'sphinx'

htmlhelp_basename = 'Sphinx-Tutorialdoc'
latex_elements = {
}
latex_documents = [
    (master_doc, 'Sphinx-Tutorial.tex', 'Sphinx-Tutorial Documentation',
     'Sooftware', 'manual'),
]



# -- Options for HTML output -------------------------------------------------

# The theme to use for HTML and HTML Help pages.  See the documentation for
# a list of builtin themes.
#
html_theme = 'sphinx_rtd_theme'
html_logo = "pyAPEP_logo.png"
# Add any paths that contain custom static files (such as style sheets) here,
# relative to this directory. They are copied after the builtin static files,
# so a file named "default.css" will overwrite the builtin "default.css".
html_static_path = ['_static']

man_pages = [
    (master_doc, 'sphinx-tutorial', 'Sphinx-Tutorial Documentation',
     [author], 1)
]
texinfo_documents = [
    (master_doc, 'Sphinx-Tutorial', 'Sphinx-Tutorial Documentation',
     author, 'Sphinx-Tutorial', 'One line description of project.',
     'Miscellaneous'),
]
epub_title = project
epub_exclude_files = ['search.html']
intersphinx_mapping = {'https://docs.python.org/': None}
todo_include_todos = True
