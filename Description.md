# Detail Parameter Settings / Default Setting
Extra parameters of listed methods other than population size / number of solutions and maximum number of iterations


## Differential Evolution (DE)
* DE contains 2 extra parameters
```code 
CR = 0.9    # crossover rate
F  = 0.5    # constant factor
opts = {'k':k, 'fold':fold, 'N':N, 'T':T, 'CR':CR, 'F':F}
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

