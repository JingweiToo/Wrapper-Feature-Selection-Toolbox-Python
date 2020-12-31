#[2010]-"A new metaheuristic bat-inspired algorithm"

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
    fmax   = 2      # maximum frequency
    fmin   = 0      # minimum frequency
    alpha  = 0.9    # constant
    gamma  = 0.9    # constant
    A_max  = 2      # maximum loudness
    r0_max = 1      # maximum pulse rate
    
    N        = opts['N']
    max_iter = opts['T']
    if 'fmax' in opts:
        fmax   = opts['fmax'] 
    if 'fmin' in opts:
        fmin   = opts['fmin'] 
    if 'alpha' in opts:
        alpha  = opts['alpha'] 
    if 'gamma' in opts:
        gamma  = opts['gamma'] 
    if 'A' in opts:
        A_max  = opts['A'] 
    if 'r' in opts:
        r0_max = opts['r'] 
        
    # Dimension
    dim = np.size(xtrain, 1)
    if np.size(lb) == 1:
        ub = ub * np.ones([1, dim], dtype='float')
        lb = lb * np.ones([1, dim], dtype='float')
        
    # Initialize position & velocity
    X     = init_position(lb, ub, N, dim)
    V     = np.zeros([N, dim], dtype='float')
    
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
    print("Best (BA):", curve[0,t])
    t += 1
    
    # Initial loudness [1 ~ 2] & pulse rate [0 ~ 1]
    A  = np.random.uniform(1, A_max, N)
    r0 = np.random.uniform(0, r0_max, N)
    r  = r0
    
    while t < max_iter:  
        Xnew  = np.zeros([N, dim], dtype='float') 
        
        for i in range(N):
            # beta [0 ~1]
            beta = rand()
            # frequency (2)
            freq = fmin + (fmax - fmin) * beta
            for d in range(dim):
                # Velocity update (3)
                V[i,d]    = V[i,d] + (X[i,d] - Xgb[0,d]) * freq 
                # Position update (4)
                Xnew[i,d] = X[i,d] + V[i,d]
                # Boundary
                Xnew[i,d] = boundary(Xnew[i,d], lb[0,d], ub[0,d])
                
            # Generate local solution around best solution
            if rand() > r[i]:
                for d in range (dim):
                    # Epsilon in [-1,1]
                    eps       = -1 + 2 * rand()
                    # Random walk (5)
                    Xnew[i,d] = Xgb[0,d] + eps * np.mean(A)              
                    # Boundary
                    Xnew[i,d] = boundary(Xnew[i,d], lb[0,d], ub[0,d])
            
        # Binary conversion
        Xbin = binary_conversion(Xnew, thres, N, dim)
        
        # Greedy selection
        for i in range(N):
            Fnew = Fun(xtrain, ytrain, Xbin[i,:], opts)
            if rand() < A[i]  and  Fnew <= fit[i,0]:
                X[i,:]   = Xnew[i,:]
                fit[i,0] = Fnew
                # Loudness update (6)
                A[i]     = alpha * A[i]
                # Pulse rate update (6)
                r[i]     = r0[i] * (1 - np.exp(-gamma * t))               
                
            if fit[i,0] < fitG:
                Xgb[0,:] = X[i,:]
                fitG     = fit[i,0]
             
        # Store result
        curve[0,t] = fitG.copy()
        print("Generation:", t + 1)
        print("Best (BA):", curve[0,t])
        t += 1            

            
    # Best feature subset
    Gbin       = binary_conversion(Xgb, thres, 1, dim) 
    Gbin       = Gbin.reshape(dim)
    pos        = np.asarray(range(0, dim))    
    sel_index  = pos[Gbin == 1]
    num_feat   = len(sel_index)
    # Create dictionary
    ba_data = {'sf': sel_index, 'c': curve, 'nf': num_feat}
    
    return ba_data  


            
            
                