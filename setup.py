# -*- coding: utf-8 -*-

"""
Setup script.
"""

from setuptools import setup, find_packages

# Package meta-data.
VERSION = 1.0
DESCRIPTION = 'Generative flower image.'

# Package abstract dependencies
REQUIRES = [
      'numpy == 1.17.0',
      'pip == 19.3.1',
      'scipy == 1.2.0',
      'tensorflow == 2.0.0',
      'matplotlib == 3.0.2',
      'pandas == 0.24.0',
      'scikit-image == 0.16.2'
]

# Long description of the package
with open("README.md", "r") as f:
    LONG_DESCRIPTION = f.read()

# A list of classifiers to categorize the project (only used for searching and
# browsing projects on PyPI).
CLASSIFIERS = [
      'Development Status :: 3 - Alpha',
      'Intended Audience :: Science/Research',
      'Intended Audience :: Developers',
      'Topic :: Software Development',
      'Topic :: Scientific/Engineering',
      'Topic :: Computer Vision',
      'Operating System :: Unix',
      'Operating System :: MacOS',
      'Programming Language :: Python',
      'Programming Language :: Python :: 3.7.0',
      'License :: OSI Approved :: MIT License'
]

# Setup
setup(name='flower-garden',
      version=VERSION,
      description=DESCRIPTION,
      long_description=LONG_DESCRIPTION,
      long_description_content_type="text/markdown",
      author='Arthur Imbert',
      author_email='arthur.imbert.pro@gmail.com',
      url='https://github.com/Henley13/flower-garden',
      packages=find_packages(),
      license='MIT',
      python_requires='==3.7.0',
      install_requires=REQUIRES,
      classifiers=CLASSIFIERS
      )
