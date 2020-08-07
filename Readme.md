<h1>XPySom</h1>

Self Organizing Maps
--------------------

XPySom is a minimalistic implementation of the Self Organizing Maps (SOM) that can seamlessly leverage vector/matrix operations made available on Numpy or CuPy, resulting in an efficient implementation for both multi-core CPUs and GP-GPUs. XPySom has been realized as a quite invasive modification to the MiniSom code available at: https://github.com/JustGlowing/minisom.git.

SOM is a type of Artificial Neural Network able to convert complex, nonlinear statistical relationships between high-dimensional data items into simple geometric relationships on a low-dimensional display.

Installation
---------------------

Download XPySom to a directory of your choice and use the setup script:

    git clone https://github.com/Manciukic/xpysom.git
    python setup.py install

How to use it
---------------------

The module interface is similar to [MiniSom](https://github.com/JustGlowing/minisom.git). In the following only the basics of the usage are reported, for an overview of all the features, please refer to the original MiniSom examples you can refer to: https://github.com/JustGlowing/minisom/tree/master/examples (you can find the same examples also in this repository but they have not been updated yet).

In order to use XPySom you need your data organized as a Numpy matrix where each row corresponds to an observation or as list of lists like the following:

```python
data = [[ 0.80,  0.55,  0.22,  0.03],
        [ 0.82,  0.50,  0.23,  0.03],
        [ 0.80,  0.54,  0.22,  0.03],
        [ 0.80,  0.53,  0.26,  0.03],
        [ 0.79,  0.56,  0.22,  0.03],
        [ 0.75,  0.60,  0.25,  0.03],
        [ 0.77,  0.59,  0.22,  0.03]]      
```

 Then you can train XPySom just as follows:

```python
from xpysom import XPySom    
som = XPySom(6, 6, 4, sigma=0.3, learning_rate=0.5) # initialization of 6x6 SOM
som.train(data, 100) # trains the SOM with 100 iterations
```

You can obtain the position of the winning neuron on the map for a given sample as follows:

```
som.winner(data[0])
```

Differences with MiniSom
---------------------
 - The batch SOM algorithm is used (instead of the online used in MiniSom). Therefore, use only `train` to train the SOM, `train_random` and `train_batch` are not present.
 - `decay_function` input parameter is no longer a function but one of `'linear'`,
 `'exponential'`, `'asymptotic'`. As a consequence of this change, `sigmaN` and `learning_rateN` have been added as input parameters to represent the values at the last iteration.
 - New input parameter `std_coeff`, used to calculate gaussian exponent denominator `d = 2*std_coeff**2*sigma**2`. Default value is 0.5 (as in [Somoclu](https://github.com/peterwittek/somoclu), which is **different from MiniSom original value** sqrt(pi)).
 - New input parameter `xp` (default = `cupy` module). Back-end to use for computations.
 - New input parameter `n_parallel` to set size of the mini-batch (how many input samples to elaborate at a time).
 - **Hexagonal** grid support is **experimental** and is significantly slower than rectangular grid.  

Additional documentation
---------------------
A publication about the design and performance of XPySom has been accepted for presentation at the [IEEE 32nd International Symposium on Computer Architecture and High Performance Computing](https://sbac2020.dcc.fc.up.pt/):
  -  Riccardo Mancini, Antonio Ritacco, Giacomo Lanciano and Tommaso Cucinotta. "XPySom: High-Performance Self-Organizing Maps," IEEE 32nd International Symposium on Computer Architecture and High Performance Computing, September 8-11, 2020. Porto, Portugal (turned to a virtual on-line event due to the Covid-19 emergency).


TODO
---------------------

 - [ ] Update examples in `examples/`
 - [ ] Improve hexagonal grid support

Compatibility notes
---------------------
XPySom has been tested under Python 3.7.6 with CuPy 7.4.0 or Numpy 1.18.1.

License
---------------------

XPySom by Riccardo Mancini, a modification of the original MiniSom by Giuseppe Vettigli, is licensed under the Creative Commons Attribution 3.0 Unported License. To view a copy of this license, visit [http://creativecommons.org/licenses/by/3.0/](http://creativecommons.org/licenses/by/3.0/ "http://creativecommons.org/licenses/by/3.0/").

![License]( http://i.creativecommons.org/l/by/3.0/88x31.png "Creative Commons Attribution 3.0 Unported License")
