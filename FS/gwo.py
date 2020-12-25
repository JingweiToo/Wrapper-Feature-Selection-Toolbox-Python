#[2014]-"Grey wolf optimizer"

import numpy as np
from numpy.random import rand
from FS.functionHO import Fun


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
    

def jfs(xtrain, ytrain, opts):
    # Parameters
    ub    = 1
    lb    = 0
    thres = 0.5
    
    N        = opts['N']
    max_iter = opts['T']
    
    # Dimension
    dim = np.size(xtrain, 1)
    if np.size(lb) == 1:
        ub = ub * np.ones([1, dim], dtype='int')
        lb = lb * np.ones([1, dim], dtype='int')
        
    # Initialize position & velocity
    X     = init_position(lb, ub, N, dim)
    
    # Binary conversion
    Xbin  = binary_conversion(X, thres, N, dim)
    
    # Fitness at first iteration
    fit    = np.zeros([N, 1], dtype='float')
    Falpha = float('inf')
    Fbeta  = float('inf')
    Fdelta = float('inf')
    for i in range(N):
        fit[i,0] = Fun(xtrain, ytrain, Xbin[i,:], opts)
        if fit[i,0] < Falpha:
            Xalpha = X[i,:]
            Falpha = fit[i,0]
            Gbin   = Xbin[i,:]
        if fit[i,0] < Fbeta and fit[i,0] > Falpha:
            Xbeta  = X[i,:]
            Fbeta  = fit[i,0]
        if fit[i,0] < Fdelta and fit[i,0] > Fbeta and fit[i,0] > Falpha:
            Xdelta = X[i,:]
            Fdelta = fit[i,0]
    
    # Pre
    curve = np.zeros([1, max_iter], dtype='float') 
    t     = 0
    
    curve[0,t] = Falpha
    print("Iteration:", t + 1)
    print("Best (GWO):", curve[0,t])
    t += 1
    
    while t < max_iter:  
      	# Coefficient decreases linearly from 2 to 0 
        a = 2 - t * (2 / max_iter) 
        for i in range(N):
            for d in range(dim):
                # Parameter C (3.4)
                C1 = 2 * rand()
                C2 = 2 * rand()
                C3 = 2 * rand()
                # Compute Dalpha, Dbeta & Ddelta (3.5)
                Dalpha = abs(C1 * Xalpha[d] - X[i,d]) 
                Dbeta  = abs(C2 * Xbeta[d] - X[i,d])
                Ddelta = abs(C3 * Xdelta[d] - X[i,d])
                # Parameter A (3.3)
                A1 = 2 * a * rand() - a
                A2 = 2 * a * rand() - a
                A3 = 2 * a * rand() - a
                # Compute X1, X2 & X3 (3.6) 
                X1 = Xalpha[d] - A1 * Dalpha
                X2 = Xbeta[d] - A2 * Dbeta
                X3 = Xdelta[d] - A3 * Ddelta
                # Update wolf (3.7)
                X[i,d] = (X1 + X2 + X3) / 3                
                # Boundary
                X[i,d] = boundary(X[i,d], lb[0,d], ub[0,d])
        
        # Binary conversion
        Xbin  = binary_conversion(X, thres, N, dim)
        
        # Fitness
        for i in range(N):
            fit[i,0] = Fun(xtrain, ytrain, Xbin[i,:], opts)
            if fit[i,0] < Falpha:
                Xalpha = X[i,:]
                Falpha = fit[i,0]
                Gbin   = Xbin[i,:]
            if fit[i,0] < Fbeta and fit[i,0] > Falpha:
                Xbeta  = X[i,:]
                Fbeta  = fit[i,0]
            if fit[i,0] < Fdelta and fit[i,0] > Fbeta and fit[i,0] > Falpha:
                Xdelta = X[i,:]
                Fdelta = fit[i,0]
        
        curve[0,t] = Falpha
        print("Iteration:", t + 1)
        print("Best (GWO):", curve[0,t])
        t += 1
    
                
    # Best feature subset
    pos        = np.asarray(range(0, dim))    
    sel_index  = pos[Gbin == 1]
    num_feat   = len(sel_index)
    # Create dictionary
    gwo_data = {'sf': sel_index, 'c': curve, 'nf': num_feat}
    
    return gwo_data 
        
                
                
                
    
