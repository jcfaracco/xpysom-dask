#!/usr/bin/env python
from distutils.core import setup

description = 'Minimalistic implementation of batch Self Organizing Maps (SOM)'
keywords = ['machine learning', 'neural networks', 'clustering', 'dimentionality reduction']

setup(name='XPySom',
      version='1.0.0',
      description=description,
      author='Riccardo Mancini',
      package_data={'': ['Readme.md']},
      include_package_data=True,
      license="CC BY 3.0",
      py_modules=['xpysom'],
      requires=['numpy', 'cupy'],
      url='https://github.com/Manciukic/xpysom',
      download_url='https://github.com/Manciukic/xpysom/archive/master.zip',
      keywords=keywords)
