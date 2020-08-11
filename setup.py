#!/usr/bin/env python
from distutils.core import setup

description = 'Minimalistic implementation of batch Self Organizing Maps (SOM) for parallel execution on CPU or GPU.'
keywords = ['machine learning', 'neural networks', 'clustering', 'dimentionality reduction']

setup(name='XPySom',
      version='1.0.0',
      description=description,
      author='Riccardo Mancini',
      author_email="r.mancini@santannapisa.it",
      package_data={'': ['Readme.md']},
      include_package_data=True,
      license="GNU General Public License v3.0",
      packages=['xpysom'],
      install_requires=['numpy'],
      extra_requires={'gpu': ['cupy']},
      url='https://github.com/Manciukic/xpysom',
      download_url='https://github.com/Manciukic/xpysom/archive/master.zip',
      keywords=keywords,
      classifiers=[
            'Development Status :: 3 - Alpha',      
            'Intended Audience :: Science/Research',    
            'License :: OSI Approved :: GNU General Public License v3 (GPLv3)',
            'Programming Language :: Python :: 3',
            'Programming Language :: Python :: 3.4',
            'Programming Language :: Python :: 3.5',
            'Programming Language :: Python :: 3.6',
            'Programming Language :: Python :: 3.7',
            'Programming Language :: Python :: 3.8',
  ],
)
