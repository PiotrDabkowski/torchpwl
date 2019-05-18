try:
    from setuptools import setup
except ImportError:
    from distutils.core import setup

long_desc = '''
Piecewise Linear Functions (PWLs) can be used to approximate any 1D function. 
PWLs are built with a configurable number of line segments - the more segments the more accurate the approximation.
This package implements PWLs in PyTorch and as such they can be fit to the data using standard gradient descent.
For example:

import torchpwl

# Create a PWL consisting of 3 segments for 5 features - each feature will have its own PWL function.
pwl = torchpwl.PWL(num_features=5, num_breakpoints=3)
x = torch.Tensor(11, 5).normal_()
y = pwl(x)


Monotonicity is also supported via `MonoPWL`. See the class documentations for more details.
'''

# rm -rf dist build && python setup.py sdist bdist_wheel
# twine upload dist/*
setup(
    name='torchpwl',
    version='0.1.0',
    packages=['torchpwl'],
    url='https://github.com/PiotrDabkowski/torchpwl',
    install_requires=['torch>=1.1.0'],
    license='MIT',
    author='Piotr Dabkowski',
    author_email='piodrus@gmail.com',
    description=
    'Implementation of Piecewise Linear Functions (PWL) in PyTorch.',
    long_description=long_desc)
