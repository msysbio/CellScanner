
# Configuration file for the Sphinx documentation builder.
#
# This file only contains a selection of the most common options. For a full
# list see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Path setup --------------------------------------------------------------

import os
import sys
sys.path.insert(0, os.path.abspath('./extensions'))

# -- Project information -----------------------------------------------------

# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information

project = "μGrowthDB"
organization = "Lab of Microbial Systems Biology"
author = f"{organization} & Contributors"
copyright = f"2025, {author}"
version = "0.0.1"
release = version

# -- General configuration ---------------------------------------------------

# Add any Sphinx extension module names here, as strings. They can be
# extensions coming with Sphinx (named 'sphinx.ext.*') or your custom
# ones.
extensions = [

    # For using CONTRIBUTING.md.
    "myst_parser",

    # Local packages.
    "youtube",
    "trello",
    "variables",
    "tags",
    "links",
    "hacks",
    "notfound.extension",

    # These extensions require RTDs to work so they will not work locally.
    "sphinx_search.extension",
    "autoapi.extension",

    # To link to pyqt5 docs
    "sphinx_qt_documentation",
    "sphinx.ext.autodoc",
    "sphinx.ext.intersphinx",
    "sphinx.ext.autosummary",
    "sphinx.ext.viewcode",
    "sphinx.ext.napoleon",
]

# -- Options for autoapi -------------------------------------------------------
autoapi_type = "python"
autoapi_dirs = ["../cellscanner"]

# List of patterns, relative to source directory, that match files and
# directories to ignore when looking for source files.
# This pattern also affects html_static_path and html_extra_path.
exclude_patterns = [
    '_build',
    'Thumbs.db',
    '.DS_Store',
    ".env",
    "extensions",
    "**/includes",
    "README.md",
    "design-tabs.js", # We are using inline-tabs and this throws errors/warnings
]

# Enable typehints
autodoc_typehints = "signature"

# Napoleon settings
napoleon_numpy_docstring = True

# The master toctree document.
master_doc = "index"

pygments_style = "sphinx"


# Add any paths that contain templates here, relative to this directory.
templates_path = ['_templates']


# -- Features ----------------------------------------------------------------

# Auto numbering of figures
numfig = True

# -- Options for HTML output -------------------------------------------------

# The theme to use for HTML and HTML Help pages.  See the documentation for
# a list of builtin themes.
# https://github.com/pradyunsg/furo
# https://pradyunsg.me/furo/
html_theme = 'furo'
html_title = "CellScanner"
html_short_title = "CellScanner"
html_logo = '_static/logo.png'
html_favicon = '_static/favicon.ico'

html_theme_options = {
    "light_logo": 'logo.png',  # "crest-oceanrender-logo.svg",
    "dark_logo": 'logo-dark.png',  # "crest-oceanrender-logo-dark.svg",
    "sidebar_hide_name": True,
    # "announcement": "<em>Important</em> announcement!",
}

html_context = {
    "organization": organization,
}

html_show_sphinx = False

# Add any paths that contain custom static files (such as style sheets) here,
# relative to this directory. They are copied after the builtin static files,
# so a file named "default.css" will overwrite the builtin "default.css".
html_static_path = ["_static"]

# These paths are either relative to html_static_path or fully qualified paths (eg. https://...).
# Increment query parameter to invalidate the cache.
html_css_files = [
    'custom.css',
]

html_js_files = [
    'https://cdnjs.cloudflare.com/ajax/libs/medium-zoom/1.0.6/medium-zoom.min.js',
    'https://p.trellocdn.com/embed.min.js',
    'custom.js',
]

html_output_encoding = "utf-8"


mathjax_path = (
    "https://cdn.mathjax.org/mathjax/latest/"
    "MathJax.js?config=TeX-AMS-MML_HTMLorMML"
)

# -- Options for PDF output --------------------------------------------------

# Customise PDF here. maketitle overrides the cover page.
latex_elements = {
    # "maketitle": "\\input{your_cover.tex}"
    # "maketitle": "\\sphinxmaketitle",
}

# latex_logo = "../logo/crest-oceanrender-logomark512.png"
latex_logo = "_static/logo.png"

# -- Templating --------------------------------------------------------------

# The default role will be used for `` so we do not need to do :get:``.
default_role = "get"


# -- Options for markdown -------------------------------------------------------

# No need to manually register .md, as myst_parser handles it
source_suffix = {
    '.rst': 'restructuredtext',
    '.md': 'markdown',  # This is registered automatically by myst_parser
}

intersphinx_mapping = {
    "python": ("https://docs.python.org/3", None),
    "PyQt5.QtWidgets": ("https://www.riverbankcomputing.com/static/Docs/PyQt5", None),
    "PySide6": ("https://doc.qt.io/qtforpython/", None),
    "Numpy": ("https://docs.scipy.org/doc/numpy/", None),
    "pandas": ("https://pandas.pydata.org/pandas-docs/stable/", None),
    "packaging": ("https://packaging.pypa.io/en/latest/", None),
}