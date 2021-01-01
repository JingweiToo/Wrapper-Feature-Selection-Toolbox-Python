#[2012]-"Flower pollination algorithm for global optimization"

import numpy as np
from numpy.random import rand
from FS.functionHO import Fun
import math


def init_position(lb, ub, N, dim):
    X = np.zeros([N, dim], dtype='float')
    for i in range(N):
        for d in range(dim):
            X[i,d] = lb[0,d] + (ub[0,d] - lb[0,d]) * rand()        
    
    return X


def binary_conversion(X, thres, N, dim):
    Xbin = np.zeros([N, dim], dtype='int')
    for i in range(N):
        for d in range(dim):
            if X[i,d] > thres:
                Xbin[i,d] = 1
            else:
                Xbin[i,d] = 0
    
    return Xbin


def boundary(x, lb, ub):
    if x < lb:
        x = lb
    if x > ub:
        x = ub
    
    return x
    

# Levy Flight
def levy_distribution(beta, dim):
    # Sigma     
    nume  = math.gamma(1 + beta) * np.sin(np.pi * beta / 2)
    deno  = math.gamma((1 + beta) / 2) * beta * 2 ** ((beta - 1) / 2)
    sigma = (nume / deno) ** (1 / beta) 
    # Parameter u & v 
    u     = np.random.randn(dim) * sigma
    v     = np.random.randn(dim)
    # Step 
    step  = u / abs(v) ** (1 / beta)
    LF    = 0.01 * step
    
    return LF


def jfs(xtrain, ytrain, opts):
    # Parameters
    ub     = 1
    lb     = 0
    thres  = 0.5
    beta   = 1.5    # levy component
    P      = 0.8    # switch probability
    
    N        = opts['N']
    max_iter = opts['T']
    if 'P' in opts:
        P    = opts['P'] 
    if 'beta' in opts:
        beta = opts['beta'] 
        
    # Dimension
    dim = np.size(xtrain, 1)
    if np.size(lb) == 1:
        ub = ub * np.ones([1, dim], dtype='float')
        lb = lb * np.ones([1, dim], dtype='float')
        
    # Initialize position 
    X     = init_position(lb, ub, N, dim)
    
    # Binary conversion
    Xbin  = binary_conversion(X, thres, N, dim)
    
    # Fitness at first iteration
    fit   = np.zeros([N, 1], dtype='float')
    Xgb   = np.zeros([1, dim], dtype='float')
    fitG  = float('inf')
    
    for i in range(N):
        fit[i,0] = Fun(xtrain, ytrain, Xbin[i,:], opts)
        if fit[i,0] < fitG:
            Xgb[0,:] = X[i,:]
            fitG     = fit[i,0]
    
    # Pre
    curve = np.zeros([1, max_iter], dtype='float') 
    t     = 0
    
    curve[0,t] = fitG.copy()
    print("Generation:", t + 1)
    print("Best (FPA):", curve[0,t])
    t += 1
    
    while t < max_iter:  
        Xnew  = np.zeros([N, dim], dtype='float') 
        
        for i in range(N):
            # Global pollination 
            if rand() < P:
                # Levy distribution (2)
                L = levy_distribution(beta, dim) 
                for d in range(dim):
                    # Global pollination (1)
                    Xnew[i,d] = X[i,d] + L[d] * (X[i,d] - Xgb[0,d]) 
                    # Boundary
                    Xnew[i,d] = boundary(Xnew[i,d], lb[0,d], ub[0,d])
                    
            # Local pollination
            else:
                # Different flower j, k in same species
                R   = np.random.permutation(N) 
                J   = R[0] 
                K   = R[1]
                # Epsilon [0 to 1]
                eps = rand()
                for d in range(dim):
                    # Local pollination (3)
                    Xnew[i,d] = X[i,d] + eps * (X[J,d] - X[K,d])           
                    # Boundary
                    Xnew[i,d] = boundary(Xnew[i,d], lb[0,d], ub[0,d])
                
        # Binary conversion
        Xbin = binary_conversion(Xnew, thres, N, dim)
        
        # Greedy selection
        for i in range(N):
            Fnew = Fun(xtrain, ytrain, Xbin[i,:], opts)
            if Fnew <= fit[i,0]:
                X[i,:]   = Xnew[i,:]
                fit[i,0] = Fnew            
                
            if fit[i,0] < fitG:
                Xgb[0,:] = X[i,:]
                fitG     = fit[i,0]
             
        # Store result
        curve[0,t] = fitG.copy()
        print("Generation:", t + 1)
        print("Best (FPA):", curve[0,t])
        t += 1            

            
    # Best feature subset
    Gbin       = binary_conversion(Xgb, thres, 1, dim) 
    Gbin       = Gbin.reshape(dim)
    pos        = np.asarray(range(0, dim))    
    sel_index  = pos[Gbin == 1]
    num_feat   = len(sel_index)
    # Create dictionary
    fpa_data = {'sf': sel_index, 'c': curve, 'nf': num_feat}
    
    return fpa_data  


            
            
                