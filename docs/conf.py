# ARTS Documentation — Sphinx configuration
# https://www.sphinx-doc.org/en/master/usage/configuration.html

import os
import pathlib

# -- Project information -----------------------------------------------------
project = "ARTS"
copyright = "2019, Battelle Memorial Institute"
author = "Pacific Northwest National Laboratory"
version = "2.0"
release = "2.0.0"

# -- General configuration ---------------------------------------------------
extensions = [
    "breathe",
    "sphinx_copybutton",
    "myst_parser",
]

templates_path = ["_templates"]
exclude_patterns = ["_build", "Thumbs.db", ".DS_Store"]

# -- Breathe (Doxygen XML bridge) --------------------------------------------
# The XML directory is populated by the 'doxygen' CMake target.
# When building via CMake: build/docs/doxygen/xml
# When building standalone: docs/_build/doxygen/xml
breathe_projects = {
    "ARTS": os.environ.get(
        "BREATHE_DOXYGEN_XML_DIR",
        str(pathlib.Path(__file__).parent / "_build" / "doxygen" / "xml"),
    ),
}
breathe_default_project = "ARTS"
breathe_domain_by_extension = {"h": "c"}
breathe_default_members = ("members", "undoc-members")

# -- MyST-Parser (Markdown support) ------------------------------------------
source_suffix = {
    ".rst": "restructuredtext",
    ".md": "markdown",
}
myst_enable_extensions = [
    "colon_fence",
    "deflist",
]

# -- HTML output (Furo theme) ------------------------------------------------
html_theme = "furo"
html_static_path = ["_static"]
html_title = "ARTS Documentation"

html_theme_options = {
    "light_css_variables": {
        "color-brand-primary": "#003366",  # PNNL navy blue
        "color-brand-content": "#003366",
    },
    "dark_css_variables": {
        "color-brand-primary": "#6699cc",
        "color-brand-content": "#6699cc",
    },
    "source_repository": "https://github.com/pnnl/ARTS",
    "source_branch": "master",
    "source_directory": "docs/",
}

# -- Options for copy-button -------------------------------------------------
copybutton_prompt_text = r"^\$ "
copybutton_prompt_is_regexp = True
