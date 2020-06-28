<h1>XPySom</h1>

Self Organizing Maps
--------------------

XPySom is a minimalistic implementation of the Self Organizing Maps (SOM) that can seamlessly leverage vector/matrix operations made available on Numpy or CuPy, resulting in an efficient implementation for both multi-core CPUs and GP-GPUs. XPySom has been realized as a quite invasive modification to the MiniSom code available at: https://github.com/JustGlowing/minisom.git.

SOM is a type of Artificial Neural Network able to convert complex, nonlinear statistical relationships between high-dimensional data items into simple geometric relationships on a low-dimensional display.

Updates about XPySom are posted on <a href="https://twitter.com/JustGlowing">Twitter</a>.

Installation
---------------------

Download XPySom to a directory of your choice and use the setup script:

    git clone https://github.com/Manciukic/xpysom.git
    python setup.py install

How to use it
---------------------

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

For an overview of all the features implemented in XPySom, please refer to the original MiniSom examples you can browse at: https://github.com/JustGlowing/minisom/tree/master/examples

#### Export a SOM and load it again

A model can be saved using pickle as follows

```python
import pickle
som = XPySom(7, 7, 4)

# ...train the som here

# saving the som in the file som.p
with open('som.p', 'wb') as outfile:
    pickle.dump(som, outfile)
```

and can be loaded as follows

```python
with open('som.p', 'rb') as infile:
    som = pickle.load(infile)
```

Note that if a lambda function is used to define the decay factor XPySom will not be pickable anymore.

Compatibility notes
---------------------
XPySom has been tested under Python 3.6.2.

License
---------------------

XPySom by Riccardo Mancini, a modification of the original MiniSom by Giuseppe Vettigli, is licensed under the Creative Commons Attribution 3.0 Unported License. To view a copy of this license, visit [http://creativecommons.org/licenses/by/3.0/](http://creativecommons.org/licenses/by/3.0/ "http://creativecommons.org/licenses/by/3.0/").

![License]( http://i.creativecommons.org/l/by/3.0/88x31.png "Creative Commons Attribution 3.0 Unported License")
