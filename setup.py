from setuptools import setup
from Cython.Build import cythonize
import numpy
setup(
    name='impact_simulation',
    version='1.0.0',    
    description='impact_simulation',
    url='https://github.com/moritzgubler/vc-sqnm',
    author='Marco Krummenacher, Jonas Finkler, Moritz Guber',
    packages=['impact_simulation'],
    ext_modules = cythonize("impact_simulation/lenjon.pyx"),
    include_dirs=[numpy.get_include()],
    install_requires=[
                    'numba',
                    'numpy', 
                    'p5',
                    'scipy',
                    'scikit-image',
                    'Cython',
                    "tqdm"
                      ]
)
