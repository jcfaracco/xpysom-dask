#!/usr/bin/env python
from setuptools import setup

# read the contents of your README file
from os import path
this_directory = path.abspath(path.dirname(__file__))
with open(path.join(this_directory, 'Readme.md'), encoding='utf-8') as f:
    long_description = f.read()

description = 'Minimalistic implementation of batch Self Organizing Maps (SOM) for parallel execution on CPU or GPU.'
keywords = ['machine learning', 'neural networks', 'clustering', 'dimentionality reduction']

setup(name='XPySom',
      version='1.0.5',
      description=description,
      long_description=long_description,
      long_description_content_type="text/markdown",
      author='Riccardo Mancini',
      author_email="r.mancini@santannapisa.it",
      package_data={'': ['Readme.md']},
      include_package_data=True,
      license="GNU General Public License v3.0",
      packages=['xpysom'],
      install_requires=['numpy'],
      extras_require={
            'cuda90': ['cupy-cuda90'],
            'cuda92': ['cupy-cuda92'],
            'cuda100': ['cupy-cuda100'],
            'cuda101': ['cupy-cuda101'],
            'cuda102': ['cupy-cuda102'],
      },
      url='https://github.com/Manciukic/xpysom',
      download_url='https://github.com/Manciukic/xpysom/archive/v1.0.5.tar.gz',
      keywords=keywords,
      classifiers=[
            'Development Status :: 4 - Beta',      
            'Intended Audience :: Science/Research',    
            'License :: OSI Approved :: GNU General Public License v3 (GPLv3)',
            'Programming Language :: Python :: 3',
            'Programming Language :: Python :: 3.4',
            'Programming Language :: Python :: 3.5',
            'Programming Language :: Python :: 3.6',
            'Programming Language :: Python :: 3.7',
            'Programming Language :: Python :: 3.8',
      ],
      zip_safe=False,
)
