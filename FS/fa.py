#[2010]-"Firefly algorithm,stochastic test functions and design optimization" 

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
    ub     = 1
    lb     = 0
    thres  = 0.5
    alpha  = 1       # constant
    beta0  = 1       # light amplitude
    gamma  = 1       # absorbtion coefficient
    theta  = 0.97    # control alpha
    
    N          = opts['N']
    max_iter   = opts['T']
    if 'alpha' in opts:
        alpha  = opts['alpha'] 
    if 'beta0' in opts:
        beta0  = opts['beta0'] 
    if 'gamma' in opts:
        gamma  = opts['gamma'] 
    if 'theta' in opts:
        theta  = opts['theta'] 
        
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
    print("Best (FA):", curve[0,t])
    t += 1
        
    while t < max_iter:  
        # Alpha update
        alpha = alpha * theta
        # Rank firefly based on their light intensity
        ind   = np.argsort(fit, axis=0)
        FF    = fit.copy()
        XX    = X.copy()
        for i in range(N):
            fit[i,0] = FF[ind[i,0]]
            X[i,:]   = XX[ind[i,0],:]
        
        for i in range(N):
            # The attractiveness parameter
            for j in range(N):
                # Update moves if firefly j brighter than firefly i
                if fit[i,0] > fit[j,0]: 
                    # Compute Euclidean distance 
                    r    = np.sqrt(np.sum((X[i,:] - X[j,:]) ** 2))
                    # Beta (2)
                    beta = beta0 * np.exp(-gamma * r ** 2)
                    for d in range(dim):
                        # Update position (3)
                        eps    = rand() - 0.5
                        X[i,d] = X[i,d] + beta * (X[j,d] - X[i,d]) + alpha * eps 
                        # Boundary
                        X[i,d] = boundary(X[i,d], lb[0,d], ub[0,d])

                    # Binary conversion
                    temp      = np.zeros([1, dim], dtype='float')
                    temp[0,:] = X[i,:]  
                    Xbin      = binary_conversion(temp, thres, 1, dim)
                    
                    # fitness
                    fit[i,0]  = Fun(xtrain, ytrain, Xbin[0,:], opts)
                    
                    # best update        
                    if fit[i,0] < fitG:
                        Xgb[0,:] = X[i,:]
                        fitG     = fit[i,0]
                
        # Store result
        curve[0,t] = fitG.copy()
        print("Generation:", t + 1)
        print("Best (FA):", curve[0,t])
        t += 1            

            
    # Best feature subset
    Gbin       = binary_conversion(Xgb, thres, 1, dim) 
    Gbin       = Gbin.reshape(dim)
    pos        = np.asarray(range(0, dim))    
    sel_index  = pos[Gbin == 1]
    num_feat   = len(sel_index)
    # Create dictionary
    fa_data = {'sf': sel_index, 'c': curve, 'nf': num_feat}
    
    return fa_data  


            
            
                