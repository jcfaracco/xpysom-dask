<h1>XPySom-Dask</h1>

Self-Organizing Maps with Dask Support
--------------------------------------

XPySom-Dask is a [Dask](https://www.dask.org/) version of the original [XPySom](https://github.com/Manciukic/xpysom) project.
The original project is a batched version of the SOM algorithm, it can be easily transformed into a distributed version using Dask.

Installation
------------

You can download XPySom-Dask from PyPi:

    pip install xpysom-dask

By default, dependencies for GPU execution are not downloaded. 
You can also specify a CUDA version to automatically download those 
requirements. For example, for CUDA Toolkit 10.2 you would write:

    pip install xpysom-dask[cuda102]

Alternatively, you can manually install XPySom-Dask.
Download XPySom to a directory of your choice and use the setup script:

    pip3 install git+https://github.com/jcfaracco/xpysom-dask.git

How to use it
-------------

The module interface is similar to [MiniSom](https://github.com/JustGlowing/minisom.git). In the following only the basics of the usage are reported, for an overview of all the features, please refer to the original MiniSom examples you can refer to: https://github.com/JustGlowing/minisom/tree/master/examples (you can find the same examples also in this repository but they have not been updated yet).

To use XPySom-Dask you need your data organized as a Dask Array matrix where each row corresponds to an observation or as a list of lists like the following:

```python
chunks = (4, 2)
data = [[ 0.80,  0.55,  0.22,  0.03],
        [ 0.82,  0.50,  0.23,  0.03],
        [ 0.80,  0.54,  0.22,  0.03],
        [ 0.80,  0.53,  0.26,  0.03],
        [ 0.79,  0.56,  0.22,  0.03],
        [ 0.75,  0.60,  0.25,  0.03],
        [ 0.77,  0.59,  0.22,  0.03]]      
```

 Then you can train XPySom-Dask just as follows:

```python
from xpysom-dask import XPySom

import dask.array as da

from dask.distributed import Client, LocalCluster

client = Client(LocalCluster())

dask_data = da.from_array(data, chunks=chunks)

som = XPySom(6, 6, 4, sigma=0.3, learning_rate=0.5, use_dask=True, chunks=chunks) # initialization of 6x6 SOM
som.train(dask_data, 100) # trains the SOM with 100 iterations
```

You can obtain the position of the winning neuron on the map for a given sample as follows:

```
som.winner(data[0])
```

Differences with MiniSom
------------------------

 - The batch SOM algorithm is used (instead of the online used in MiniSom). Therefore, use only `train` to train the SOM, `train_random` and `train_batch` are not present.
 - `decay_function` input parameter is no longer a function but one of `'linear'`,
 `'exponential'`, `'asymptotic'`. As a consequence of this change, `sigmaN` and `learning_rateN` have been added as input parameters to represent the values at the last iteration.
 - New input parameter `std_coeff`, used to calculate gaussian exponent denominator `d = 2*std_coeff**2*sigma**2`. Default value is 0.5 (as in [Somoclu](https://github.com/peterwittek/somoclu), which is **different from MiniSom original value** sqrt(pi)).
 - New input parameter `xp` (default = `cupy` module). Back-end to use for computations.
 - New input parameter `n_parallel` to set size of the mini-batch (how many input samples to elaborate at a time).
 - **Hexagonal** grid support is **experimental** and is significantly slower than rectangular grid.  


Cite
----

If you are using this project in your research, please cite the paper where XPySom-Dask.

```bibtex
@inproceedings{dasf,
  title        = {DASF: a high-performance and scalable framework for large seismic datasets},
  author       = {Julio C. Faracco and Otávio O. Napoli and João Seródio and Carlos A. Astudillo and Leandro Villas and Edson Borin and Alan A. Souza and Daniel C. Miranda and João Paulo Navarro},
  year         = {2024},
  month        = {August},
  booktitle    = {Proceedings of the International Meeting for Applied Geoscience and Energy},
  address      = {Houston, TX},
  organization = {AAPG/SEG}
}
```

Authors
-------

Copyright (C) 2021 Julio Faracco
