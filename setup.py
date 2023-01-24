from setuptools import setup

setup(
    name='impact_simulation',
    version='1.0.0',
    description='impact_simulation',
    url='https://github.com/moritzgubler/vc-sqnm',
    author='Marco Krummenacher, Jonas Finkler, Moritz Guber',
    packages=['impact_simulation'],
    install_requires=[
                    'numba',
                    'numpy',
                    'p5',
                    'scipy',
                    'scikit-image',
                    'tqdm',
                    'pyqtgraph'
                      ]
)

