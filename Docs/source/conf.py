import os
import sys

import matplotlib

matplotlib.use("Agg")

sys.path.insert(0, os.path.abspath("../../"))

# -- Project information --
project = "Bayesian Deep Learning Library"
copyright = "2025, Aatu Selkee"
author = "Aatu Selkee"
release = "0.1.0"

# -- General configuration --
extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.viewcode",
    "sphinx.ext.napoleon",
    "sphinx.ext.mathjax",
    "sphinx_gallery.gen_gallery",
]

modindex_common_prefix = ["BayesianDLL."]
add_module_names = True
templates_path = ["_templates"]
exclude_patterns = []
autodoc_member_order = "bysource"

# -- Options for HTML output --
html_theme = "sphinx_rtd_theme"
html_static_path = ["_static"]
html_css_files = ["custom.css"]
html_theme_options = {
    "collapse_navigation": False,
    "sticky_navigation": True,
    "navigation_depth": 2,
}

# -- Sphinx-Gallery configuration --
sphinx_gallery_conf = {
    "examples_dirs": ["../../examples"],  # example scripts
    "gallery_dirs": ["auto_examples"],  # where the generated gallery is written
    "filename_pattern": r".*\.py$",  # execute every example
    "ignore_pattern": r".*NO_DOCS_.*\.py$",  # prefix a file with NO_DOCS_ to leave it out
    "image_scrapers": ("matplotlib",),
    "plot_gallery": True,
    "backreferences_dir": "gen_modules/backreferences",
    "doc_module": ("BayesianDLL",),
    "reference_url": {"BayesianDLL": None},
    "within_subsection_order": "FileNameSortKey",
}
