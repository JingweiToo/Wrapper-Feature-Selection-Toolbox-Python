# Jx-WFST : Wrapper Feature Selection Toolbox

[![License](https://img.shields.io/badge/license-BSD_3-blue.svg)](https://github.com/JingweiToo/Wrapper-Feature-Selection-Toolbox-Python/blob/main/LICENSE)
[![GitHub release](https://img.shields.io/badge/release-pre-yellow.svg)](https://github.com/JingweiToo/Wrapper-Feature-Selection-Toolbox-Python)

---
> "Toward Talent Scientist: Sharing and Learning Together"
>  --- [Jingwei Too](https://jingweitoo.wordpress.com/)
---

![Wheel](https://www.mathworks.com/matlabcentral/mlc-downloads/downloads/5dc2bdb4-ce4b-4e0e-bd6e-0237ff6ddde1/f9a9e760-64b9-4e31-9903-dffcabdf8be6/images/1607601518.JPG)


## Introduction

* This toolbox offers 13 wrapper feature selection methods
* The `Demo_PSO` provides an example of how to apply PSO on benchmark dataset 
* Source code of these methods are written based on pseudocode & paper


## Usage
The main function `jfs` is adopted to perform feature selection. You may switch the algorithm by changing the `pso` in `from FS.pso import jfs` to [other abbreviations](/README.md#list-of-available-wrapper-feature-selection-methods)
* If you wish to use particle swarm optimization ( PSO ) then you may write
```code
from FS.pso import jfs
```
* If you want to use differential evolution ( DE ) then you may write
```code
from FS.de import jfs
```


## Input
* *`feat`*   : feature vector matrix ( Instance *x* Features )
* *`label`*  : label matrix ( Instance *x* 1 )
* *`opts`*   : parameter settings
    + *`N`* : number of solutions / population size ( *for all methods* )
    + *`T`* : maximum number of iterations ( *for all methods* )
    + *`k`* : *k*-value in *k*-nearest neighbor 


## Output
* *`Acc`*  : accuracy of validation model
* *`fmdl`* : feature selection model ( It contains several results )
    + *`sf`* : index of selected features
    + *`nf`* : number of selected features
    + *`c`*  : convergence curve
    
    
### Example 1 : Particle Swarm Optimization ( PSO ) 
```code 
import numpy as np
import pandas as pd
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from FS.pso import jfs   # change this to switch algorithm 
import matplotlib.pyplot as plt


# load data
data  = pd.read_csv('ionosphere.csv')
data  = data.values
feat  = np.asarray(data[:, 0:-1])   # feature vector
label = np.asarray(data[:, -1])     # label vector

# split data into train & validation (70 -- 30)
xtrain, xtest, ytrain, ytest = train_test_split(feat, label, test_size=0.3, stratify=label)
fold = {'xt':xtrain, 'yt':ytrain, 'xv':xtest, 'yv':ytest}

# parameter
k    = 5     # k-value in KNN
N    = 10    # number of particles
T    = 100   # maximum number of iterations
w    = 0.9
c1   = 2
c2   = 2
opts = {'k':k, 'fold':fold, 'N':N, 'T':T, 'w':w, 'c1':c1, 'c2':c2}

# perform feature selection
fmdl = jfs(feat, label, opts)
sf   = fmdl['sf']

# model with selected features
num_train = np.size(xtrain, 0)
num_valid = np.size(xtest, 0)
x_train   = xtrain[:, sf]
y_train   = ytrain.reshape(num_train)  # Solve bug
x_valid   = xtest[:, sf]
y_valid   = ytest.reshape(num_valid)  # Solve bug

mdl       = KNeighborsClassifier(n_neighbors = k) 
mdl.fit(x_train, y_train)

# accuracy
y_pred    = mdl.predict(x_valid)
Acc       = np.sum(y_valid == y_pred)  / num_valid
print("Accuracy:", 100 * Acc)

# number of selected features
num_feat = fmdl['nf']
print("Feature Size:", num_feat)

# plot convergence
curve   = fmdl['c']
curve   = curve.reshape(np.size(curve,1))
x       = np.arange(0, opts['T'], 1.0) + 1.0

fig, ax = plt.subplots()
ax.plot(x, curve, 'o-')
ax.set_xlabel('Number of Iterations')
ax.set_ylabel('Fitness')
ax.set_title('PSO')
ax.grid()
plt.show()
```

### Example 2 : Genetic Algorithm ( GA ) 
```code 
import numpy as np
import pandas as pd
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from FS.ga import jfs   # change this to switch algorithm 
import matplotlib.pyplot as plt


# load data
data  = pd.read_csv('ionosphere.csv')
data  = data.values
feat  = np.asarray(data[:, 0:-1])
label = np.asarray(data[:, -1])

# split data into train & validation (70 -- 30)
xtrain, xtest, ytrain, ytest = train_test_split(feat, label, test_size=0.3, stratify=label)
fold = {'xt':xtrain, 'yt':ytrain, 'xv':xtest, 'yv':ytest}

# parameter
k    = 5     # k-value in KNN
N    = 10    # number of chromosomes
T    = 100   # maximum number of generations
CR   = 0.8
MR   = 0.01
opts = {'k':k, 'fold':fold, 'N':N, 'T':T, 'CR':CR, 'MR':MR}

# perform feature selection
fmdl = jfs(feat, label, opts)
sf   = fmdl['sf']

# model with selected features
num_train = np.size(xtrain, 0)
num_valid = np.size(xtest, 0)
x_train   = xtrain[:, sf]
y_train   = ytrain.reshape(num_train)  # Solve bug
x_valid   = xtest[:, sf]
y_valid   = ytest.reshape(num_valid)  # Solve bug

mdl       = KNeighborsClassifier(n_neighbors = k) 
mdl.fit(x_train, y_train)

# accuracy
y_pred    = mdl.predict(x_valid)
Acc       = np.sum(y_valid == y_pred)  / num_valid
print("Accuracy:", 100 * Acc)

# number of selected features
num_feat = fmdl['nf']
print("Feature Size:", num_feat)

# plot convergence
curve   = fmdl['c']
curve   = curve.reshape(np.size(curve,1))
x       = np.arange(0, opts['T'], 1.0) + 1.0

fig, ax = plt.subplots()
ax.plot(x, curve, 'o-')
ax.set_xlabel('Number of Iterations')
ax.set_ylabel('Fitness')
ax.set_title('GA')
ax.grid()
plt.show()
```


## Requirement

* Python 3 
* Numpy
* Pandas
* Scikit-learn
* Matplotlib


## List of available wrapper feature selection methods
* Note that the methods are altered so that they can be used in feature selection tasks 
* The extra parameters represent the parameter(s) other than population size and maximum number of iterations
* Click on the name of method to view how to set the extra parameter(s)
* Use the *`opts`* to set the specific parameters
* If you do not set extra parameters then the algorithm will use default setting in [here](/Description.md)


| No. | Abbreviation | Name                                                                                        | Year | Extra Parameters |
|-----|--------------|---------------------------------------------------------------------------------------------|------|------------------|
| 13  | `hho`        | Harris Hawk Optimization                                                                    | 2019 | No               |
| 12  | `ssa`        | Salp Swarm Algorithm                                                                        | 2017 | No               |
| 11  | `woa`        | [Whale Optimization Algorithm](/Description.md#whale-optimization-algorithm-woa)            | 2016 | Yes              |
| 10  | `sca`        | [Sine Cosine Algorithm](/Description.md#sine-cosine-algorithm-sca)                          | 2016 | Yes              |
| 09  | `ja`         | Jaya Algorithm                                                                              | 2016 | No               |
| 08  | `gwo`        | Grey Wolf Optimizer                                                                         | 2014 | No               |
| 07  | `fpa`        | [Flower Pollination Algorithm](/Description.md#flower-pollination-algorithm-fpa)            | 2012 | Yes              |
| 06  | `ba`         | [Bat Algorithm](/Description.md#bat-algorithm-ba)                                           | 2010 | Yes              |
| 05  | `fa`         | [Firefly Algorithm](/Description.md#firefly-algorithm-fa)                                   | 2010 | Yes              |
| 04  | `cs`         | [Cuckoo Search Algorithm](/Description.md#cuckoo-search-cs)                                 | 2009 | Yes              |
| 03  | `de`         | [Differential Evolution](/Description.md#differential-evolution-de)                         | 1997 | Yes              |
| 02  | `pso`        | [Particle Swarm Optimization](/Description.md#particle-swarm-optimization-pso)              | 1995 | Yes              |
| 01  | `ga`         | [Genetic Algorithm](/Description.md#genetic-algorithm-ga)                                   | -    | Yes              |



