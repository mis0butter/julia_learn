# define packages 
using Revise 
using GP_june 
using LinearAlgebra 
using ForwardDiff 


## ============================================ ## 
# GENERATE TRAINING DATA 

# true hyperparameters 
sig_f0 = 1  
l0     = 1 
sig_n0 = 0.1 

# generate training data 
N = 5 
# x_train = sort( 10*rand(N,1), dims=1 )      # matrix     
x_train = [ 1.0343 2.8932 4.1403 5.1443 5.3743 ] 

# did everything work 
test = sq_dist(x_train, x_train) 

# covariance function from kernel (squared exponential) 
k_fn((sig_f, l, xp, xq)) = sig_f^2 * exp.( -1/( 2*l^2 ) * sq_dist(xp, xq) ) 

sigma = k_fn( sig_f0, l0, x_train, x_train ) 
sigma += sig_n0^2 * I 

# test gauss_sample 
y = gauss_sample(0*x_train, sigma, 1) 


## ============================================ ## 
# PRIOR DISTRIBUTION (based on created test data points) 
# INIT: no idea what idea data looks like 

# generate test data x-star = test data 
dx = 0.1 
x_test = collect(0 : dx : 10.0) 
x_test = reshape(x_test, length(x_test), 1)

# mean function 
mu_fn(x) = 0 .* x 
mu_prior = mu_fn(x_test) 

# prior covariance 
Kss = k_fn(sig_f0, l0, x_test, x_test) 
S_prior = Kss 

# fs_prior = gauss_sample(mu_prior, S_prior, 1) 
fs_prior = gauss_sample(mu_prior, S_prior + sig_n0^2 * I , 1) 

## ============================================ ##
# POSTERIOR DISTRIBUTION (based on training data) 

K = k_fn(sig_f0, l0, x_train, x_train) 
K += sig_n0^2 * I 
Ks = k_fn(sig_f0, l0, x_train, x_test) 

# conditional (posterior) distribution 
mu_post = Ks' * inv(K) * y 
S_post  = Kss - ( Ks' * inv(K) *  Ks )  
det(S_post + sig_n0^2 * I) 
tr( S_post + sig_n0^2 * I )

# fs_post = gauss_sample(mu_post, S_post, 1)
# fs_post = gauss_sample(mu_post, S_post + sig_n0^2 * I , 1)


## ============================================ ## 

# log-likelihood function 
function log_Z( (dX, Θ, Ξ, sig_f, l, sig_n) )

    # covariance function from kernel (squared exponential) 
    k_fn(sig_f, l, xp, xq) = sig_f^2 * exp.( -1/( 2*l^2 ) * sq_dist(xp, xq) ) 
    println("k_fn") 

    Ky = k_fn(sig_f, l, Ξ, Ξ) 
    println("Ky") 

    # log-likelihood 
    term1 = 1/2 * ( dX - Θ * Ξ )' * inv(Ky) * ( dX - Θ * Ξ )
    term2 = 1/2 * log(Ky) 
    term3 = 1/2 * log(2π)
    println("terms")
    
    log_Z_out = term1 + term2 + term3 
    # logZ_out = Ky 

    return log_Z_out 

end 



# log-likelihood function 
function log_Z2( (dX, Θ, Ξ, sig_f, l, sig_n) ) 

    # covariance function from kernel (squared exponential) 
    k_fn(sig_f, l, xp, xq) = sig_f^2 * exp.( -1/( 2*l^2 ) * sq_dist(xp, xq) ) 

    Ky = k_fn(sig_f, l, Ξ, Ξ) 

    # # log-likelihood 
    # term1 = 1/2 * ( dX - Θ * Ξ )' * inv(Ky) * ( dX - Θ * Ξ )
    # term2 = 1/2 * log(Ky) 
    # term3 = n/2 * log(2π)
    
    # log_Z_out = term1 + term2 + term3 
    logZ_out = Ky 

    return log_Z_out 

end 

## ============================================ ##
# SCRATCH CODE 

function test_fn( (a, b, c, d) ) 

    out = a + b^2 + c^3 + d^4 

    return out 

end 

f_xyz( (x,y,z) ) = 5sin(x*y) + 2*y/4z 

