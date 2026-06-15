"""
Sphinx configuration for KernelGenBench documentation.

Build command:

   $ make html  # build HTML documentation
"""

import os
import sys

# -- Path setup --------------------------------------------------------------

# If extensions (or modules to document with autodoc) are in another directory,
# add these directories to sys.path here.

# -- Project information -----------------------------------------------------

project = "KernelGenBench Documentation"
copyright = '2026, FlagOS Community'
author = 'FlagOS Community'
release = '1.0.0'

# -- General configuration ---------------------------------------------------

extensions = [
    "myst_parser",
    "sphinx_copybutton",
    "sphinx_design",
    "sphinx.ext.autodoc",
    "sphinx.ext.autosectionlabel",
    "sphinx.ext.extlinks",
    "sphinx.ext.intersphinx",
    "sphinxext.opengraph",
    "sphinx_tippy",
    "sphinx_togglebutton",
]

templates_path = ["../_templates"]
exclude_patterns = ["_build", "_includes"]
master_doc = "index"
default_role = "obj"

# MyST Parser configuration
myst_enable_extensions = [
    "dollarmath",
    "amsmath",
    "deflist",
    "fieldlist",
    "html_admonition",
    "html_image",
    "colon_fence",
    "smartquotes",
    "replacements",
    "strikethrough",
    "substitution",
    "tasklist",
    "attrs_inline",
    "attrs_block",
]
myst_heading_anchors = 2

# -- Internationalization ----------------------------------------------------

language = "en"
locale_dirs = ["locale/"]
gettext_compact = False  # Generate separate POT file for each source file

# -- HTML output -------------------------------------------------------------

html_theme = "sphinx_book_theme"
html_static_path = ["../_static"]
html_css_files = []
html_js_files = []

# Logo and favicon configuration based on language
# Read the Docs builds each language separately with -D language=zh_CN
if language == "zh_CN":
    html_logo = "../_static/images/logos/zh-logo.svg"
    html_theme_options = {
        "home_page_in_toc": True,
        "use_download_button": False,
        "repository_url": "https://github.com/flagos-ai/KernelGenBench",
        "use_edit_page_button": True,
        "use_repository_button": True,
        "logo": {
            "image_light": "../_static/images/logos/zh-logo.svg",
            "image_dark": "../_static/images/logos/zh-logo-dark.svg",
        },
    }
else:
    # Default to English
    html_logo = "../_static/images/logos/en-logo.svg"
    html_theme_options = {
        "home_page_in_toc": True,
        "use_download_button": False,
        "repository_url": "https://github.com/flagos-ai/KernelGenBench",
        "use_edit_page_button": True,
        "use_repository_button": True,
        "logo": {
            "image_light": "../_static/images/logos/en-logo.svg",
            "image_dark": "../_static/images/logos/en-logo-dark.svg",
        },
    }

html_favicon = "../_static/images/logos/favicon.svg"

htmlhelp_basename = "KernelGenBenchdoc"

# -- LaTeX output ------------------------------------------------------------

latex_documents = [
    (
        "index",
        "KernelGenBench.tex",
        "KernelGenBench Documentation",
        "KernelGenBench Team",
        "manual",
    ),
]

# -- Manual page output ------------------------------------------------------

man_pages = [
    (
        "index",
        "kernelgenbench",
        "KernelGenBench Documentation",
        ["KernelGenBench Team"],
        1,
    )
]

# -- OpenGraph ---------------------------------------------------------------

ogp_site_name = "KernelGenBench Documentation"
ogp_use_first_image = True
ogp_enable_meta_description = True
ogp_description_length = 300

# -- Intersphinx -------------------------------------------------------------

intersphinx_cache_limit = 14
intersphinx_timeout = 3
intersphinx_mapping = {
    "python": ("https://docs.python.org/3.10/", None),
    "sphinx": ("https://www.sphinx-doc.org/en/master/", None),
}
intersphinx_disabled_reftypes = ["*"]

# -- Linkcheck ---------------------------------------------------------------

linkcheck_retries = 2
linkcheck_timeout = 1
linkcheck_workers = 10
linkcheck_ignore = [
    r"http://127\.0\.0\.1",
    r"http://localhost",
    r"https://github\.com.+?#L\d+",
]

# -- Extensions configuration -------------------------------------------------

autosectionlabel_prefix_document = True

extlinks = {
    "issue": ("https://github.com/flagos-ai/KernelGenBench/issues/%s", "#%s"),
}

suppress_warnings = ["epub.unknown_project_files"]
