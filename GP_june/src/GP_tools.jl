using LinearAlgebra 


## ============================================ ##
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


## ============================================ ##
# sample from given mean and covariance 

export k_fn 

function k_fn(( σ_f, l, xp, xq ))

    return σ_f^2 * exp.( -1/( 2*l^2 ) * sq_dist(xp, xq) ) 

end 


## ============================================ ##
# define square distance function 

export sq_dist 

# define square distance function 
function sq_dist(a::Vector, b::Vector) 

    r = length(a) 
    p = length(b) 

    # iterate 
    C = zeros(r,p) 
    for i = 1:r 
        for j = 1:p 
            C[i,j] = ( a[i] - b[j] )^2 
        end 
    end 

    return C 

end 


## ============================================ ##
# marginal log-likelihood for Gaussian Process 

export log_p 

function log_p(( σ_f, l, σ_n, x, y, μ ))
    
    # training kernel function 
    Ky = k_fn((σ_f, l, x, x)) 
    Ky += σ_n^2 * I 

    term_1 = 1/2 * ( y - μ )' * inv( Ky ) * ( y - μ ) 
    term_2 = 1/2 * log( det( Ky ) ) 

    return term_1 + term_2 

end 


## ============================================ ##
# posterior distribution 

export post_dist

function post_dist(( x_train, y_train, x_test, σ_f, l, σ_n ))

    # x  = training data  
    # xs = test data 
    # joint distribution 
    #   [ y  ]     (    [ K(x,x)+Ïƒ_n^2*I  K(x,xs)  ] ) 
    #   [ fs ] ~ N ( 0, [ K(xs,x)         K(xs,xs) ] ) 

    # covariance from training data 
    K    = k_fn((σ_f, l, x_train, x_train))  
    K   += σ_n^2 * I       # add noise for positive definite 
    Ks   = k_fn((σ_f, l, x_train, x_test))  
    Kss  = k_fn((σ_f, l, x_test, x_test)) 

    # conditional distribution 
    # mu_cond    = K(Xs,X)*inv(K(X,X))*y
    # sigma_cond = K(Xs,Xs) - K(Xs,X)*inv(K(X,X))*K(X,Xs) 
    # fs | (Xs, X, y) ~ N ( mu_cond, sigma_cond ) 
    μ_post = Ks' * K^-1 * y_train 
    Σ_post = Kss - (Ks' * K^-1 * Ks)  

    return μ_post, Σ_post

end 

