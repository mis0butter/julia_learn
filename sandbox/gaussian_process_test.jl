using Optim 
using LinearAlgebra
using Statistics 
using Plots 

## ============================================ ##
# functions 

# define square distance function 
function sq_dist(a::Vector, b::Vector) 

    # transform into matrix 
    a = a[:,:] 
    b = b[:,:]

    # extract dims 
    D, n = size(a)   
    d, m = size(b) 

    # ensure a, b are "row" matrices 
    if D > n 
        a = transpose(a)  
        D, n = size(a) 
    end 
    if d > m 
        b = transpose(b)  
        d, m = size(b) 
    end 

    # iterate 
    C = zeros(n,m) 
    for d = 1:D 
        amat = repeat(transpose(a), outer = [1,m]) 
        bmat = repeat(b, outer = [n,1]) 
        C += (bmat - amat).^2 
    end 

    return C 

end 

# sample from given mean and covariance 
function gauss_sample(mu, K::Matrix) 

    # ensure is matrix 
    mu = mu[:,:] 

    # ENSURE MU IS COLUMN 
    r, p = size(mu) 
    if p > r 
        mu = transpose(mu) 
        r, p = size(mu) 
    end 
    
    # cholesky decomposition, get lower triangular decomp 
    C = cholesky(K) 
    L = C.L 

    # draw random samples 
    u = randn(length(mu)) 

    # fs ~ N(0, K(xs, xs)) 
    f = mu + L*u 
    f = f[:] 

    return f 

end 

## ============================================ ##
# GP !!!  

# true hyperparameters 
σ_f0 = 1.0 ;    σ_f = σ_f0 
l_0  = 1.0 ;    l   = l_0 
σ_n0 = 0.1 ;    σ_n = σ_n0 

# generate training data 
N = 100 
x_train = sort( 10*rand(N) )      

# kernel function 
k_fn(σ_f, l, xp, xq) = σ_f^2 * exp.( -1/( 2*l^2 ) * sq_dist(xp, xq) ) 

Σ_train = k_fn( σ_f0, l_0, x_train, x_train ) 
Σ_train += σ_n0^2 * I 

# training data --> "measured" output at x_train 
y_train = gauss_sample(0*x_train, Σ_train) 

## ============================================ ##
# generate test points 

# generate test points 
x_test = 0 : 0.1 : 10 
x_test = collect(x_test) 

# mean function of 0 
meanZero = 0 * x_test

# "prior" / test covariance  
Kss = k_fn( σ_f0, l_0, x_test, x_test ) 

# posterior distribution 
K = k_fn(σ_f0, l_0, x_train, x_train) 
K += σ_n0^2 * I 

# covariance from training / measuremenet data AND test data 
Ks = k_fn(σ_f0, l_0, x_train, x_test) 

# conditional distribution / PREDICTION 
μ_post = Ks' * inv(K) * y_train 
Σ_post = Kss - ( Ks' * inv(K) * Ks )


## ============================================ ##
# marginal log-likelihood for Gaussian Process 
function log_p(( σ_f, l, σ_n, y, x ))
    
    # kernel function 
    k_fn(σ_f, l, xp, xq) = σ_f^2 * exp.( -1/( 2*l^2 ) * sq_dist(xp, xq) ) 

    # training kernel function 
    Ky = k_fn(σ_f, l, x, x) 
    Ky += σ_n^2 * I 

    term = zeros(2)
    term[1] = 1/2*( y )'*inv( Ky )*( y ) 
    term[2] = 1/2*log(det( Ky )) 

    return sum(term)

end 

# test log-likelihood function 
log_p(( σ_f, l, σ_n, y_train, x_train ))

## ============================================ ##
# solve for hyperparameters
# log_p(( σ_f, l, σ_n )) = log_p(( σ_f, l, σ_n, y_train, x_train ))

# test reassigning function 
test_log_p(( σ_f, l, σ_n )) = log_p(( σ_f, l, σ_n, y_train, x_train ))
test_log_p(( σ_f, l, σ_n ))

# σ_0    = [σ_f0, l_0, σ_n0] 
σ_0    = [ σ_f, l_0, σ_n ] * 1.1

result = optimize(test_log_p, σ_0) 
println("log_p min = ", result.minimizer) 


