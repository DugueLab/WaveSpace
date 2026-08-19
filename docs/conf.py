# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information
import os
import sys
from importlib.metadata import version as _pkg_version

sys.path.insert(0, os.path.abspath(".."))

# Mark that we are building docs (used by examples to skip slow sections)
os.environ["BUILDING_DOCS"] = "1"

# Provide data directory path to examples (Sphinx-Gallery exec doesn't have __file__)
os.environ["WAVEDATA_EXAMPLE_DIR"] = os.path.normpath(
    os.path.join(os.path.dirname(__file__), "..", "Examples", "ExampleData")
)

# Ensure matplotlib uses non-interactive backend for doc builds
import matplotlib

matplotlib.use("Agg")

# Create ExampleData symlink in auto_examples so __file__-relative paths work
# when Sphinx-Gallery copies scripts into docs/auto_examples/
_examples_dir = os.path.join(os.path.dirname(__file__), "..", "Examples")
_auto_examples_dir = os.path.join(os.path.dirname(__file__), "auto_examples")
_target = os.path.join(_examples_dir, "ExampleData")
_link = os.path.join(_auto_examples_dir, "ExampleData")
if os.path.isdir(_target):
    os.makedirs(_auto_examples_dir, exist_ok=True)
    if not os.path.exists(_link):
        os.symlink(_target, _link)

project = "WaveSpace"
copyright = "2025-2026, Kirsten Petras, Dennis Croonenberg, Laura Dugué"
author = "Kirsten Petras, Dennis Croonenberg, Laura Dugué"
release = _pkg_version("WaveSpace")

# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration

extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.autosummary",
    "sphinx.ext.napoleon",
    "sphinx.ext.intersphinx",
    "sphinx.ext.viewcode",
    "sphinx_gallery.gen_gallery",
]

sphinx_gallery_conf = {
    "examples_dirs": "../Examples",
    "gallery_dirs": "auto_examples",
    "plot_gallery": True,
    "filename_pattern": r"WSExample",
    "remove_config_comments": True,
}

intersphinx_mapping = {
    "numpy": ("https://numpy.org/doc/stable/", None),
    "scipy": ("https://docs.scipy.org/doc/scipy/", None),
    "matplotlib": ("https://matplotlib.org/stable/", None),
    "pandas": ("https://pandas.pydata.org/docs/", None),
    "mne": ("https://mne.tools/stable/", None),
}

autosummary_generate = True
templates_path = ["_templates"]
exclude_patterns = ["_build", "Thumbs.db", ".DS_Store"]


# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output

html_theme = "pydata_sphinx_theme"
