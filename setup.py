from setuptools import setup

setup(
    name='LennardJonesMD',
    version='1.0.0',
    description='LennardJonesMD',
    url='https://github.com/moritzgubler/LennardJonesMD.git',
    author='Marco Krummenacher, Jonas Finkler, Moritz Guber',
    packages=['LennardJonesMD'],
    install_requires=[
                    'numba',
                    'numpy',
                    'p5',
                    'scipy',
                    'scikit-image',
                    'tqdm',
                    'pyqtgraph',
                    'matplotlib',
                    'PyQt5',
                    'progressbar2',
                    'argparse',
                    ],
    entry_points={
      'console_scripts': [
        'runSimulation=LennardJonesMD.runSimulation:main',
        'makePlot=LennardJonesMD.makePlot:main',
        'makeMovie=LennardJonesMD.makeMovie:main'
      ]
    }
)

