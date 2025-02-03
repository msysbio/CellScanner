CellScanner
==================

.. only:: html

   .. image:: https://readthedocs.org/projects/crest/badge/?version=latest
      :target: https://cellscanner.readthedocs.io/en/latest/?badge=latest
      :alt: Documentation Status

.. NOTE:
.. Subsequent captions are broken in PDFs: https://github.com/sphinx-doc/sphinx/issues/4977.

.. NOTE:
.. :numbered: has bugs with PDFs: https://github.com/sphinx-doc/sphinx/issues/4318.

.. NOTE:
.. only directive does not work with tocree directive for HTML.

.. .. only:: latex
..
..    .. toctree::
..       :hidden:
..       :caption: User Guide
..
..       about/introduction

.. NOTE:
.. ":numbered: 1" means numbering is only one deep. Needed for the version history.


.. toctree::
   :numbered: 1
   :maxdepth: 2
   :caption: About CellScanner

   about/background
   about/known-issues
   about/history
..    about/integrations


.. toctree::
   :numbered: 
   :maxdepth: 3
   :caption: Tutorial

   tutorials/gui


.. toctree::
   :numbered: 
   :maxdepth: 3
   :caption: Tips and hints

   faqs/faqs



.. toctree::
   :maxdepth: 3
   :caption: Developer Guide

   .. dev/contributing


.. ===============================================



.. NOTE:
.. Tried to have only the title show in the ToC, but it looks like Sphinx is ignoring toctree options.

.. .. only:: latex
..
..    .. toctree::
..
..       meta/history

.. TODO:
..   user/support

.. only:: html

   .. .. toctree::
   ..    :maxdepth: 3
   ..    :caption: Developer Guide

   ..    dev/contributing

