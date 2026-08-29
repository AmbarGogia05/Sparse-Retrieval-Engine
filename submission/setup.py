"""
Build definition for the optional compiled extensions (pybind11).

CI / grading runs this from INSIDE submission/:

    python setup.py build_ext --inplace

so the sources and module names below are relative to submission/. The
resulting submission/_vbyte_cpp*.so and submission/_index_cpp*.so are
imported as submission._vbyte_cpp / submission._index_cpp. Both are
optional at runtime: indexer.py and custom_scorer.py fall back to pure
Python if the extensions aren't built, so a pure-Python submission needs
nothing here.

Build dependencies (pybind11, setuptools) are pinned in the top-level
requirements.txt. Do not commit the built .so files (see .gitignore) and
do not compile inside build_index() — the extensions are built once, here.
"""
from pybind11.setup_helpers import Pybind11Extension, build_ext
from setuptools import setup

ext_modules = [
    Pybind11Extension("_vbyte_cpp", ["_vbyte_cpp.cpp"], cxx_std=17),
    Pybind11Extension("_index_cpp", ["_index_cpp.cpp"], cxx_std=17),
    # _spimi_cpp and _stem_cpp share the tokenizer/stemmer (tokenizer.cpp); each
    # extension compiles its own copy in (separate .so, no shared linkage).
    Pybind11Extension("_spimi_cpp", ["_spimi_cpp.cpp", "tokenizer.cpp"], cxx_std=17),
    Pybind11Extension("_stem_cpp", ["_stem_cpp.cpp", "tokenizer.cpp"], cxx_std=17),
]

setup(
    name="sparse_retrieval_ext",
    ext_modules=ext_modules,
    cmdclass={"build_ext": build_ext},
)
