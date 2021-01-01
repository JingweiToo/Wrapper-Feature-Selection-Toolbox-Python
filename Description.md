# Detail Parameter Settings / Default Setting
Extra parameters of listed methods other than population size / number of solutions and maximum number of iterations

## Bat Algorithm (BA)
* BA contains 6 extra parameters
```code 
fmax   = 2      # maximum frequency
fmin   = 0      # minimum frequency
alpha  = 0.9    # constant
gamma  = 0.9    # constant
A      = 2      # maximum loudness
r      = 1      # maximum pulse rate
opts = {'k':k, 'fold':fold, 'N':N, 'T':T, 'fmax':fmax, 'fmin':fmin, 'alpha':alpha, 'gamma':gamma, 'A':A, 'r':r}
```

## Cuckoo Search (CS)
* CS contains 1 extra parameter
```code 
Pa  = 0.25   # discovery rate
opts = {'k':k, 'fold':fold, 'N':N, 'T':T, 'Pa':Pa}
```

## Differential Evolution (DE)
* DE contains 2 extra parameters
```code 
CR = 0.9    # crossover rate
F  = 0.5    # constant factor
opts = {'k':k, 'fold':fold, 'N':N, 'T':T, 'CR':CR, 'F':F}
```

## Firefly Algorithm (FA)
* FA contains 4 extra parameters
```code 
alpha  = 1       # constant
beta0  = 1       # light amplitude
gamma  = 1       # absorbtion coefficient
theta  = 0.97    # control alpha
opts = {'k':k, 'fold':fold, 'N':N, 'T':T, 'alpha':alpha, 'beta0':beta0, 'gamma':gamma, 'theta':theta}
```

## Flower Pollination Algorithm (FPA)
* FPA contains 1 extra parameter
```code 
P  = 0.8      # switch probability
opts = {'k':k, 'fold':fold, 'N':N, 'T':T, 'P':P}
```

## Genetic Algorithm (GA)
* GA contains 2 extra parameters
```code 
CR  = 0.8      # crossover rate
MR  = 0.01     # mutation rate
opts = {'k':k, 'fold':fold, 'N':N, 'T':T, 'CR':CR, 'MR':MR}
```

## Particle Swarm Optimization (PSO)
* PSO contains 3 extra parameters
```code 
c1  = 2         # cognitive factor
c2  = 2         # social factor 
w   = 0.9       # inertia weight
opts = {'k':k, 'fold':fold, 'N':N, 'T':T, 'w':w, 'c1':c1, 'c2':c2}
```

## Sine Cosine Algorithm (SCA)
* SCA contains 1 extra parameter
```code
alpha  = 2    # constant
opts = {'k':k, 'fold':fold, 'N':N, 'T':T, 'alpha':alpha}
```


## Whale Optimization Algorithm (WOA)
* WOA contains 1 extra parameter
```code
b  = 1    # constant
opts = {'k':k, 'fold':fold, 'N':N, 'T':T, 'b':b}
```


