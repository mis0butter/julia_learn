using LinearAlgebra 

# sample from given mean and covariance 
export gauss_sample 
function gauss_sample(mu::Matrix, K::Matrix, n::Int) 

    # ENSURE MU IS COLUMN 
    r, p = size(mu) ; 
    if p > r 
        mu = transpose(mu) ; 
        r, p = size(mu) ; 
    end 
    
    # cholesky decomposition, get lower triangular decomp 
    C = cholesky(K) ; 
    L = C.L ; 

    # draw random samples 
    u = randn(length(mu), n) ; 

    # fs ~ N(0, K(xs, xs)) 
    f = mu + L*u ; 

    return f 

end 