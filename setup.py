import os
import sys
from os.path import dirname

from setuptools import setup

here = os.path.abspath(dirname(__file__))

with open(os.path.join(here, 'README.rst')) as f:
    long_description = '\n' + f.read()

base_dir = os.path.dirname(__file__)

about = {}
with open(os.path.join(base_dir, "symfeat", "__version__.py")) as f:
    exec(f.read(), about)

if sys.argv[-1] == "publish":
    os.system("python setup.py sdist bdist_wheel upload")
    sys.exit()

required = [
    "numpy",
    "sympy",
    "toolz",
    "joblib",
]

setup(
    name='symfeat',
    version=about['__version__'],
    description='Ruled based feature engineering for regression',
    long_description=long_description,
    author='Markus Quade',
    author_email='info@markusqua.de',
    url='https://github.com/ohjeah/symfeat',
    packages=['symfeat'],
    install_requires=required,
    license='MIT',
    classifiers=[
        'Programming Language :: Python',
        'Programming Language :: Python :: 3.5',
        'Programming Language :: Python :: 3.6',
    ],
)
