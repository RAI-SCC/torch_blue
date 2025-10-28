# type: ignore
# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

import os
import sys

sys.path.insert(0, os.path.abspath("../../torch_bayesian"))


# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information

project = "torch_bayesian"
copyright = "2025, RAI-SCC"
author = "RAI-SCC"
release = "0.9.0"

# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration

master_doc = "index"

autodoc_typehints = "signature"
autodoc_typehints_format = "short"
python_use_unqualified_type_names = True
autodoc_type_aliases = {
    "torch.nn.common_types._size_1_t": "int | Tuple[int,]",
    "torch.nn.common_types._size_2_t": "int | Tuple[int, int]",
    "torch.nn.common_types._size_3_t": "int | Tuple[int, int, int]",
    "torch_bayesian.vi.utils.common_types.VIReturn[torch.Tensor]": "torch.Tensor | Tuple[torch.Tensor, torch.Tensor]",
    "torch_bayesian.vi.utils.common_types.VIReturn[Tuple[torch.Tensor, Optional[torch.Tensor]]]": "Tuple[torch.Tensor, Optional[torch.Tensor]] | Tuple[Tuple[torch.Tensor, torch.Tensor | None], torch.Tensor]",
    "torch_bayesian.vi.utils.common_types.VIReturn[torch.nn.common_types._tensor_list_t]": "Union[torch.Tensor, List[torch.Tensor]] | Tuple[Union[torch.Tensor, List[torch.Tensor]], torch.Tensor]",
    "torch_bayesian.vi.utils.common_types._vardist_any_t": "VariationalDistribution | List[VariationalDistribution]",
    "torch_bayesian.vi.utils.common_types._prior_any_t": "Prior | List[Prior]",
    "torch_bayesian.vi.utils.common_types._log_prob_return_format[torch.Tensor]": "Tuple[torch.Tensor, torch.Tensor]",
    "Ellipsis": "...",
    "torch_bayesian.vi.base.": "torch_bayesian.vi.",
    #    "torch.Tensor": "Tensor",
}

napoleon_numpy_docstring = True
napoleon_use_ivar = True
napoleon_preprocess_types = False
napoleon_type_aliases = autodoc_type_aliases

extensions = [
    "sphinx.ext.napoleon",
    # "sphinx.ext.inheritance_diagram",
    # "sphinx.ext.apidoc",
    "sphinx_rtd_theme",
    "sphinx.ext.viewcode",
    "sphinx.ext.autodoc",
    "autoapi.extension",
]

autoapi_type = "python"
autoapi_dirs = ["../../torch_bayesian/"]
autoapi_ignore = ["*/tests/*"]  # , "*[!_].py"]
autoapi_template_dir = "_templates/"
autoapi_own_page_level = "class"

templates_path = ["_templates"]
# exclude_patterns = []


def skip_submodules(app, what, name, obj, skip, options):
    """Skip submodules."""
    if what == "module":
        # print(f"skipping module: {name}")
        skip = True
    return skip


"""
def skip_attributes(app, what, name, obj, skip, options):
    if what == "attribute":
        print(f"skipping attribute: {name}")
        skip = True
    return skip
"""

"""
def filter_type_hints(app, what, name, obj, skip, options):
    if what == "method":
        for replaced_name in autodoc_type_aliases.keys():
            if replaced_name in obj.args:
                print(f"Before: {obj.name}: {obj.args}")
                obj.args = obj.args.replace(replaced_name, autodoc_type_aliases[replaced_name])
                print(f"After: {obj.name}: {obj.args}")
    return skip
"""

"""
def filter_type_hints(app, what, name, obj, options, lines):
    for i in range(len(lines)):

"""


def filter_type_hints(value):
    """Postprocess type hints."""
    for replaced_name in autodoc_type_aliases.keys():
        if replaced_name in value:
            value = value.replace(replaced_name, autodoc_type_aliases[replaced_name])

    # print("Filter is run")
    return value


def autoapi_prepare_jinja_env(jinja_env):
    """Modify the Jinja2 environment."""
    jinja_env.filters["custom_filter_type_hints"] = filter_type_hints


def setup(sphinx):
    """Link the custom filters to the corresponding events."""
    sphinx.connect("autoapi-skip-member", skip_submodules)
    # sphinx.connect("autoapi-skip-member", filter_type_hints)
    # sphinx.connect("autodoc-process-docstring", filter_type_hints)
    # sphinx.connect("autoapi-skip-member", skip_attributes)


# apidoc_modules = [
#    {
#        "path": "../../torch_bayesian/",
#        "destination": "../source",
#        "module_first": False,
#        "implicit_namespaces": False,
#        "separate_modules": False,
#        "exclude_patterns": ["*/test*", "*[!_].py"]#"*/vi/*.py"],
#    },
# ]

autoapi_options = [
    "members",
    #'undoc-members',
    #'private-members',
    "show-inheritance",
    "show-module-summary",
    #'special-members',
    "imported-members",
]


# add_module_names = False


# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output

html_theme = "sphinx_rtd_theme"
html_theme_options = {
    "navigation_depth": -1,
    "titles_only": True,
}

html_static_path = ["_static"]

html_css_files = [
    "css/custom.css",
]

# command line (in venv: sphinx-build -T -E -b html ./source ./build)
