

using LinearAlgebra 
using GP_june 
using SparseArrays
using Plots 

## ============================================ ##
# setup 

m = 1500            # number of examples 
n = 5000            # number of features 
p = 100/n           # sparsity density 

x0 = sprandn(n,1,p)    
A  = randn(m,n) 
B  = 1 ./ sqrt.( sum(A.^2, dims=1) )
A  = A * spdiagm(n, n, B[:])
b  = A*x0 + sqrt(0.001) * randn(m,1) 

lambda = 0.1 
x  = rand(n,1) 
z  = rand(n,1) 

p = objective(A, b, lambda, x, z) 
println("objective p = ", p)

## ============================================ ##
# lasso_admm 
    
x, hist = lasso_admm(A, b, lambda, 1.0, 1.0) 


## ============================================ ##
# plot! 

K = length(hist.objval) 

p_objval = plot(1:K, hist.objval) 
p_r_norm = plot(1:K, hist.r_norm) 
p_s_norm = plot(1:K, hist.s_norm)

p_fig = plot(p_objval, p_r_norm, p_s_norm, layout = (3,1), size = [ 600,800 ])
