using LinearAlgebra 

# sample from given mean and covariance 
export gauss_sample 

# sample from given mean and covariance 
function gauss_sample(μ::Vector, K::Matrix) 
    
    # cholesky decomposition, get lower triangular decomp 
    C = cholesky(K) ; 
    L = C.L 

    # draw random samples 
    u = randn(length(μ)) 

    # f ~ N(mu, K(x, x)) 
    f = μ + L*u

    return f 

end 

