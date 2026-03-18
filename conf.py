import os
import sys
sys.path.insert(0, os.path.abspath('.'))

extensions = [
    'sphinx.ext.autodoc',
    'sphinx.ext.napoleon',  # Supports Google/NumPy style docstrings
    'sphinx.ext.viewcode',   # Adds links to your source code
]

project = 'Your Project Name'
extensions = ['sphinx.ext.autodoc', 'sphinx.ext.napoleon']
html_theme = 'sphinx_rtd_theme'