# -*- coding: utf-8 -*-
# Author: Arthur Imbert <arthur.imbert.pro@gmail.com>
# License: BSD 3 clause

"""
Setup script.
"""

from setuptools import setup, find_packages

# Package meta-data.
DESCRIPTION = 'Repository with generative dcgan.'

# package version
VERSION = None
with open('flowers/__init__.py', encoding='utf-8') as f:
    for row in f:
        if row.startswith('__version__'):
            VERSION = row.strip().split()[-1][1:-1]
            break

# package dependencies
with open("requirements.txt", encoding='utf-8') as f:
    REQUIREMENTS = [l.strip() for l in f.readlines() if l]

# long description of the package
with open("README.md", "r") as f:
    LONG_DESCRIPTION = f.read()

# A list of classifiers to categorize the project (only used for searching and
# browsing projects on PyPI).
CLASSIFIERS = [
    'Development Status :: 3 - Alpha',
    'Intended Audience :: Science/Research',
    'Intended Audience :: Developers',
    'Intended Audience :: Data scientist',
    'Topic :: Software Development',
    'Topic :: Scientific/Engineering',
    'Topic :: Computer Vision',
    'Topic :: Generative model',
    'Topic :: Flowers',
    'Operating System :: Unix',
    'Operating System :: MacOS',
    'Programming Language :: Python',
    'Programming Language :: Python :: 3.7.0',
    'License :: OSI Approved :: MIT License'
]

# Setup
setup(name='flowers-garden',
      version=VERSION,
      description=DESCRIPTION,
      long_description=LONG_DESCRIPTION,
      long_description_content_type="text/markdown",
      author='Arthur Imbert',
      author_email='arthur.imbert.pro@gmail.com',
      url='https://github.com/Henley13/flowers-garden',
      packages=find_packages(),
      license='MIT License',
      python_requires='==3.7.0',
      install_requires=REQUIREMENTS,
      classifiers=CLASSIFIERS
      )
