#[1997]-"Differential evolution - A simple and efficient heuristic for global optimization over continuous spaces"

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
    CR    = 0.9     # crossover rate
    F     = 0.5     # factor
    
    N        = opts['N']
    max_iter = opts['T']
    if 'CR' in opts:
        CR   = opts['CR'] 
    if 'F' in opts:
        F    = opts['F']     
    
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
    print("Best (DE):", curve[0,t])
    t += 1

    while t < max_iter:  
        V = np.zeros([N, dim], dtype='float')
        U = np.zeros([N, dim], dtype='float')
        
        for i in range(N):
            # Choose r1, r2, r3 randomly, but not equal to i 
            RN = np.random.permutation(N)
            for j in range(N):
                if RN[j] == i:
                    RN = np.delete(RN, j)
                    break
                
            r1 = RN[0]
            r2 = RN[1]
            r3 = RN[2]
            # mutation (2)
            for d in range(dim):
                V[i,d] = X[r1,d] + F * (X[r2,d] - X[r3,d])
                # Boundary
                V[i,d] = boundary(V[i,d], lb[0,d], ub[0,d])
            
            # Random one dimension from 1 to dim
            index = np.random.randint(low = 0, high = dim)
            # crossover (3-4)
            for d in range(dim):
                if (rand() <= CR)  or  (d == index):
                    U[i,d] = V[i,d]
                else:
                    U[i,d] = X[i,d]
        
        # Binary conversion
        Ubin = binary_conversion(U, thres, N, dim)
        
        # Selection
        for i in range(N):
            fitU = Fun(xtrain, ytrain, Ubin[i,:], opts)
            if fitU <= fit[i,0]:
                X[i,:]   = U[i,:]
                fit[i,0] = fitU
                
            if fit[i,0] < fitG:
                Xgb[0,:] = X[i,:]
                fitG     = fit[i,0]
            
                
        # Store result
        curve[0,t] = fitG.copy()
        print("Generation:", t + 1)
        print("Best (DE):", curve[0,t])
        t += 1            

            
    # Best feature subset
    Gbin       = binary_conversion(Xgb, thres, 1, dim) 
    Gbin       = Gbin.reshape(dim)
    pos        = np.asarray(range(0, dim))    
    sel_index  = pos[Gbin == 1]
    num_feat   = len(sel_index)
    # Create dictionary
    de_data = {'sf': sel_index, 'c': curve, 'nf': num_feat}
    
    return de_data  


            
            
                