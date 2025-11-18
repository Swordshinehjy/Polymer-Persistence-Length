from setuptools import setup, Extension
from Cython.Build import cythonize
import numpy as np

extensions = [
    Extension(
        "polymer_pl.chain_rotation",
        ["src/polymer_pl/chain_rotation.pyx"],
        include_dirs=[np.get_include()],
    )
]

setup(
    ext_modules=cythonize(extensions, language_level="3"),
)


