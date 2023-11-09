# Toby DeKara and Alice Paul
# Created: Oct 19, 2021
# A python script to generate synthetic data sets
# This script is adapted from  'gen_synthetic',
# in the package 'l0bnb', which is under an MIT License, 
# copyright (c) 2020 [Hussein Hazimeh, and Rahul Mazumder, Ali Saab].

# Variable settings: 
# Bertsimas et al.  use: 
# supp_size in {5,10}, rho in {0.5, 0.8, 0.9}?
# Hazimeh et al, use n = 10^3, p = {10^3, 10^4, 10^5, 10^6}, snr = 5

import sys
import numpy as np
from numpy.random import multivariate_normal, normal
import os
import subprocess

def make_syn_data(n_mat=10**3, n=10**3, p=100, rho=0.5, snr=5, batch_n=1, seed=2022):
    """Generate a synthetic regression dataset: y, x, and b.
    The data matrix x is sampled from a multivariate gaussian with exponential 
    correlation between columns.
    The response y = xb + epsilon, where b is a vector with 'supp_size' 
    randomly chosen entries equal to 1 and the rest equal to 0.
    The error term epsilon is sampled from an normal distribution (independent
    of x). 
    
    Inputs:
        n: Number of samples, i.e. number of rows of matrix x, and length of y.
        p: Number of features, i.e. number of columns of x.
        supp_size: Number of non-zero entries in b. This is number of columns 
		of x used to construct y.
        rho: Exponential correlation parameter.
            cov_mat[row, col] = rho**np.abs(row-col)
        snr: Signal-to-noise ratio.
        batch_n: Batch number, used for records.
    Returns:
        None: x, y, and b are all saved to csv files. 
    """
    
    #xy_out_dir = f'synthetic_data/{p_sub_dir}/batch_{batch_n}'
    xy_out_dir = f'synthetic_data/batch_{batch_n}'
    os.makedirs(xy_out_dir, exist_ok=True)

    np.random.seed(seed)
    supp_size = int(0.1*p)
    support_mat = np.zeros((n_mat,supp_size))

    for i in range(n_mat):
     
        # Make x matrix
        cov_mat = np.zeros((p,p))
        for row in range(p):
            for col in range(p):
                cov_mat[row, col] = rho**np.abs(row-col)
        x = multivariate_normal(mean=np.zeros(p), cov=cov_mat, size=n)
        x_centered = x - np.mean(x, axis = 0)
        x_normalized =  x_centered / np.linalg.norm(x_centered, axis = 0)
        
        # Make y
        unshuffled_support = [i for i in range(p) if i % (p/supp_size) == 0]
        b = np.zeros((p, 1))
        b[unshuffled_support] = np.ones((supp_size,1))
        mu = np.matmul(x, b)
        var_xb = (np.std(mu, ddof=1)) ** 2
        sd_epsilon = np.sqrt(var_xb / snr)
        epsilon = normal(size=n, scale=sd_epsilon)
        y = mu + epsilon.reshape(-1,1)
        y_centered = y - np.mean(y)
        y_normalized = y_centered / np.linalg.norm(y_centered)
    
        # Shuffle x
        perm = np.random.permutation(p)
        x_shuffled = x_normalized[:, perm]

        # Identify support after shuffle
        support = [ind for ind in range(p) if perm[ind] in set(unshuffled_support)]

        # Record support
        support_mat[i] = np.array(support, ndmin=1)

        # Save x and y
        # Note: For brevity, the n value recorded in the file names is log base 10 
        # the log base 10 of the actual values
    
        # Make file name
        filetag = f'gen_syn_n{int(np.log10(n))}_p{p}_corr{rho}_snr{snr}_seed{seed}' 
        np.savetxt(f'{xy_out_dir}/x_{filetag}_{i}.csv', x_shuffled,delimiter=",")
        np.savetxt(f'{xy_out_dir}/y_{filetag}_{i}.csv', y_normalized,delimiter=",")

    # Save support for all data sets
    b_out_dir = f'synthetic_data/seed_support_records'
    os.makedirs(b_out_dir, exist_ok=True)
    np.savetxt(f'{b_out_dir}/support_corr{rho}_snr{snr}_batch{batch_n}.csv', \
               support_mat, delimiter=",")


make_syn_data(n_mat=1,n=10**3, p=10)

# if __name__ == "__main__":
# 	make_syn_data(n_mat=int(sys.argv[1]), p=int(sys.argv[2]), rho=float(sys.argv[3]), \
# 	snr=float(sys.argv[4]), batch_n = int(sys.argv[5]), seed=2022)

