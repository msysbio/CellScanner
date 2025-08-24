# Configuration file for the Sphinx documentation builder.

# -- Path setup --------------------------------------------------------------

import os
import sys

# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information

project = "CellScanner"
organization = "Lab of Microbial Systems Biology"
author = f"{organization} & Contributors"
copyright = f"2025, {author}"
version = "0.0.1"
release = version

# -- General configuration ---------------------------------------------------

sys.path.insert(0, os.path.abspath('./extensions'))

# Add any Sphinx extension module names here, as strings.
# They can be extensions coming with Sphinx (named 'sphinx.ext.*') or your custom ones.
# IMPORTANT NOTE: The order you give the extensions in the extensions list MATTERS!
# e.g. https://github.com/sphinx-doc/sphinx/issues/4221
extensions = [

    # To link to pyqt5 docs
    "sphinx.ext.autodoc",
    "sphinx.ext.intersphinx",
    "sphinx.ext.mathjax",
    "sphinx.ext.viewcode",
    "sphinx.ext.napoleon",
    "sphinx.ext.autosummary",
    "sphinx_qt_documentation",

    "nbsphinx",
    "autoapi.extension",
    "sphinx_search.extension",

    # For using CONTRIBUTING.md.
    "myst_parser",

    # Local packages.
    "youtube",
    "trello",
    "variables",
    "tags",
    "links",
    "hacks",
    # "notfound.extension",    ## not in the bac_Growt

]


# -- Options for autoapi -------------------------------------------------------
autoapi_dirs = ["../cellscanner"]
autoapi_ignore = []

# NOTE: The autoapi_options and the functions autoapi_skip_member() and setup()
# make sure class attributes are not shown on the API
autoapi_options = [
    "members",
    "undoc-members",
    "show-inheritance",
]

def autoapi_skip_member(app, what, name, obj, skip, options):
    # Skip all attributes globally
    if what == "attribute":
        return True
    return None

def setup(app):
    app.connect("autoapi-skip-member", autoapi_skip_member)

# Enable typehints
autodoc_typehints = "signature"

# Napoleon settings
napoleon_numpy_docstring = True

# The master toctree document.
master_doc = "index"

pygments_style = "sphinx"

# -- Options for HTML output -------------------------------------------------

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


# -- Features ----------------------------------------------------------------

# Auto numbering of figures
numfig = True

issues_github_path = "hariszaf/cellscanner"

mathjax_path = (
    "https://cdn.mathjax.org/mathjax/latest/"
    "MathJax.js?config=TeX-AMS-MML_HTMLorMML"
)
exclude_patterns = ['_build', 'Thumbs.db', '.DS_Store']


# Add any paths that contain templates here, relative to this directory.
templates_path = ['_templates']


# myst_enable_extensions = ["colon_fence"]

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
        # from here: https://github.com/GPflow/tensorflow-intersphinx/
    "tensorflow": ("https://www.tensorflow.org/api_docs/python", "https://github.com/GPflow/tensorflow-intersphinx/raw/master/tf2_py_objects.inv"),
    'sklearn': ('http://scikit-learn.org/stable', None)
}