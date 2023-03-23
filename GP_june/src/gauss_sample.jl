using LinearAlgebra 

# sample from given mean and covariance 
export gauss_sample 

# sample from given mean and covariance 
function gauss_sample(mu::Vector, K::Matrix) 
    
    # cholesky decomposition, get lower triangular decomp 
    C = cholesky(K) ; 
    L = C.L 

    # draw random samples 
    u = randn(length(mu)) 

    # f ~ N(mu, K(x, x)) 
    f = mu + L*u 

    return f 

end 

# test function 
C = rand(3,3)
K = C + C' + 10*I 
gauss_sample(rand(3), K)