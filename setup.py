#!/usr/bin/env python
from setuptools import setup

xpysom_version = '1.0.7.1'

# read the contents of your README file
from os import path
this_directory = path.abspath(path.dirname(__file__))
with open(path.join(this_directory, 'README.md'), encoding='utf-8') as f:
    long_description = f.read()

description = 'Minimalistic implementation of batch Self Organizing Maps (SOM) for parallel execution on CPU, GPU or Dask.'
keywords = ['machine learning', 'neural networks', 'clustering', 'dimentionality reduction']

setup(name='XPySom-dask',
      version='1.0.7',
      description=description,
      long_description=long_description,
      long_description_content_type="text/markdown",
      author='Julio Faracco',
      author_email="jcfaracco@gmail.com",
      package_data={'': ['README.md']},
      include_package_data=True,
      license="Creative Commons Attribution 3.0 Unported",
      packages=['xpysom-dask'],
      install_requires=['numpy'],
      extras_require={
            'cuda100': ['cupy-cuda100'],
            'cuda101': ['cupy-cuda101'],
            'cuda102': ['cupy-cuda102'],
            'cuda110': ['cupy-cuda110'],
            'cuda111': ['cupy-cuda111'],
            'cuda112': ['cupy-cuda112'],
            'cuda113': ['cupy-cuda113'],
            'cuda117': ['cupy-cuda117'],
            'cuda118': ['cupy-cuda118'],
      },
      url='https://github.com/jcfaracco/xpysom-dask',
      download_url=f'https://github.com/jcfaracco/xpysom-dask/archive/v{xpysom_version}.tar.gz',
      keywords=keywords,
      classifiers=[
            'Development Status :: 4 - Beta',      
            'Intended Audience :: Science/Research',    
            'License :: OSI Approved :: MIT License',
            'Programming Language :: Python :: 3',
            'Programming Language :: Python :: 3.4',
            'Programming Language :: Python :: 3.5',
            'Programming Language :: Python :: 3.6',
            'Programming Language :: Python :: 3.7',
            'Programming Language :: Python :: 3.8',
      ],
      zip_safe=False,
)
