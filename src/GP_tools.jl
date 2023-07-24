using LinearAlgebra 
using Optim 

## ============================================ ##
# sample from given mean and covariance 

export gauss_sample 
function gauss_sample(μ, K) 
# function gauss_sample(μ::Vector, K::Matrix) 

    # adding rounding ... 
    K = round.( K, digits = 10 )
    
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

export k_SE 
function k_SE( σ_f, l, xp, xq )

    K = σ_f^2 * exp.( -1/( 2*l^2 ) * sq_dist(xp, xq) )     

    # deal with det(Ky) = 0 
    # if det(K) == 0 
    #     K *= length(xp)
    # end  

    return K 

end 


## ============================================ ##
# sample from given mean and covariance 

export k_periodic
function k_periodic( σ_f, l, p, xp, xq )

    K = σ_f^2 * exp.( -2/( l^2 ) * sin.( π/p * sq_dist(xp, xq)) )     

    # deal with det(Ky) = 0 
    # if det(K) == 0 
    #     K *= length(xp)
    # end  

    return K 

end 


## ============================================ ##
# define square distance function 

export sq_dist 
function sq_dist(a, b) 
# function sq_dist(a::Vector, b::Vector) 

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
# marginal log-likelihood for Gaussian Processes  

export mlog_like 
function mlog_like( σ_f, l, σ_n, x, y, μ )
# from algorithm 2.1 of Rasmussen GP textbook  
    
    # training kernel function 
    K = k_SE(σ_f, l, x, x) 

    C = cholesky( K + σ_n^2 * I  )
    α = C.U \ ( C.L \ y ) 

    # NEGATIVE log-likelihood 
    n = length(y) 
    log_p = 1/2 * y' * α + sum( log.( diag(C.L) ) ) + n/2 * log( 2π )
    # log_p = 1/2 * y' * α + log( det(K + σ_n^2) )
    
    return log_p 

end 


## ============================================ ##
# marginal log-likelihood for Gaussian Processes 

export log_p 
function log_p( σ_f, l, σ_n, x, y, μ )
    
    # training kernel function 
    Ky = k_SE(σ_f, l, x, x) 
    Ky += σ_n^2 * I 
    
    while det(Ky) == 0 
        println( "det(Ky) = 0" )
        Ky += σ_n * I 
    end 

    term  = 1/2 * ( y - μ )' * inv( Ky ) * ( y - μ ) 
    term += 1/2 * log( det( Ky ) ) 

    return term 

end 


## ============================================ ##
# posterior distribution 

export post_dist
function post_dist( x_train, y_train, x_test, σ_f, l, σ_n )

    # x  = training data  
    # xs = test data 
    # joint distribution 
    #   [ y  ]     (    [ K(x,x) + σ_n^2*I  K(x,xs)  ] ) 
    #   [ fs ] ~ N ( 0, [ K(xs,x)           K(xs,xs) ] ) 

    # covariance from training data 
    K    = k_SE(σ_f, l, x_train, x_train)  
    Ks   = k_SE(σ_f, l, x_train, x_test)  
    Kss  = k_SE(σ_f, l, x_test, x_test) 

    # conditional distribution 
    # mu_cond    = K(Xs,X)*inv(K(X,X))*y
    # sigma_cond = K(Xs,Xs) - K(Xs,X)*inv(K(X,X))*K(X,Xs) 

    # fs | (Xs, X, y) ~ N ( mu_cond, sigma_cond ) 
    # μ_post = Ks' * K^-1 * y_train 
    # Σ_post = Kss - (Ks' * K^-1 * Ks)  

    C = cholesky(K + σ_n^2 * I) 
    α = C.U \ ( C.L \ y_train ) 
    v = C.L \ Ks 
    μ_post = Ks' * α 
    Σ_post = Kss - v'*v 

    return μ_post, Σ_post

end 

## ============================================ ##
# hp optimization (June) --> post mean  

export post_dist_hp_opt 
function post_dist_hp_opt( x_train, y_train, x_test, plot_option = false )

    # IC 
    hp = [ 1.0, 1.0, 0.1 ] 

    # optimization 
    hp_opt(( σ_f, l, σ_n )) = log_p( σ_f, l, σ_n, x_train, y_train, 0*y_train )
    od       = OnceDifferentiable( hp_opt, hp ; autodiff = :forward ) 
    result   = optimize( od, hp, LBFGS() ) 
    hp       = result.minimizer 

    μ_post, Σ_post = post_dist( x_train, y_train, x_test, hp[1], hp[2], hp[3] ) 

    if plot_option 
        p = scatter( x_train, y_train, label = "train" )
        scatter!( p, x_test, μ_post, label = "post", ls = :dash )
        display(p) 
    end 

    return μ_post, Σ_post, hp 
end 

## ============================================ ##
# posterior mean with GP toolbox 

using GaussianProcesses

export post_dist_SE 
function post_dist_SE( x_train, x_test, y_train ) 

    # kernel  
    mZero     = MeanZero() ;            # zero mean function 
    kern      = SE( 0.0, 0.0 ) ;        # squared eponential kernel (hyperparams on log scale) 
    log_noise = log(0.1) ;              # (optional) log std dev of obs noise 

    # fit GP 
    gp      = GP(x_train, y_train, mZero, kern, log_noise) 
    optimize!(gp) 
    μ, σ²   = predict_y( gp, x_test )    

    # return HPs 
    σ_f = sqrt( gp.kernel.σ2 ) 
    l   = sqrt.( gp.kernel.ℓ2 )  
    σ_n = exp( gp.logNoise.value )  
    hp  = [σ_f, l, σ_n] 

    return μ, σ², hp 
end 

export post_dist_M12A
function post_dist_M12A( x_train, x_test, y_train ) 

    # kernel  
    mZero     = MeanZero() ;            # zero mean function 
    kern      = Mat12Ard( [0.0], 0.0 ) ;        # squared eponential kernel (hyperparams on log scale) 
    log_noise = log(0.1) ;              # (optional) log std dev of obs noise 

    # fit GP 
    gp      = GP(x_train, y_train, mZero, kern, log_noise) 
    optimize!(gp) 
    μ, σ²   = predict_y( gp, x_test )    
    
    # return HPs 
    σ_f = sqrt( gp.kernel.σ2 ) 
    l   = sqrt.( gp.kernel.iℓ2[1] )  
    σ_n = exp( gp.logNoise.value )  
    hp  = [σ_f, l, σ_n] 

    return μ, σ², hp 
end 


export post_dist_M32A
function post_dist_M32A( x_train, x_test, y_train ) 

    # kernel  
    mZero     = MeanZero() ;            # zero mean function 
    kern      = Mat32Ard( [0.0], 0.0 ) ;        # squared eponential kernel (hyperparams on log scale) 
    log_noise = log(0.1) ;              # (optional) log std dev of obs noise 

    # fit GP 
    gp      = GP(x_train, y_train, mZero, kern, log_noise) 
    optimize!(gp) 
    μ, σ²   = predict_y( gp, x_test )    
    
    # return HPs 
    σ_f = sqrt( gp.kernel.σ2 ) 
    l   = sqrt.( gp.kernel.iℓ2[1] )  
    σ_n = exp( gp.logNoise.value )  
    hp  = [σ_f, l, σ_n] 

    return μ, σ², hp 
end 


export post_dist_M52A
function post_dist_M52A( x_train, x_test, y_train ) 

    # kernel  
    mZero     = MeanZero() ;            # zero mean function 
    kern      = Mat52Ard( [0.0], 0.0 ) ;        # squared eponential kernel (hyperparams on log scale) 
    log_noise = log(0.1) ;              # (optional) log std dev of obs noise 

    # fit GP 
    gp      = GP(x_train, y_train, mZero, kern, log_noise) 
    optimize!(gp) 
    μ, σ²   = predict_y( gp, x_test )    
    
    # return HPs 
    σ_f = sqrt( gp.kernel.σ2 ) 
    l   = sqrt.( gp.kernel.iℓ2[1] )  
    σ_n = exp( gp.logNoise.value )  
    hp  = [σ_f, l, σ_n] 

    return μ, σ², hp 
end 


export post_dist_M12I
function post_dist_M12I( x_train, x_test, y_train ) 

    # kernel  
    mZero     = MeanZero() ;            # zero mean function 
    kern      = Mat12Iso( 0.0, 0.0 ) ;        # squared eponential kernel (hyperparams on log scale) 
    log_noise = log(0.1) ;              # (optional) log std dev of obs noise 

    # fit GP 
    gp      = GP(x_train, y_train, mZero, kern, log_noise) 
    optimize!(gp) 
    μ, σ²   = predict_y( gp, x_test )    
    
    # return HPs 
    σ_f = sqrt( gp.kernel.σ2 ) 
    l   = gp.kernel.ℓ   
    σ_n = exp( gp.logNoise.value )  
    hp  = [σ_f, l, σ_n] 

    return μ, σ², hp 
end 


export post_dist_M32I
function post_dist_M32I( x_train, x_test, y_train ) 

    # kernel  
    mZero     = MeanZero() ;            # zero mean function 
    kern      = Mat32Iso( 0.0, 0.0 ) ;        # squared eponential kernel (hyperparams on log scale) 
    log_noise = log(0.1) ;              # (optional) log std dev of obs noise 

    # fit GP 
    gp      = GP(x_train, y_train, mZero, kern, log_noise) 
    optimize!(gp) 
    μ, σ²   = predict_y( gp, x_test )    
    
    # return HPs 
    σ_f = sqrt( gp.kernel.σ2 ) 
    l   = gp.kernel.ℓ   
    σ_n = exp( gp.logNoise.value )  
    hp  = [σ_f, l, σ_n] 

    return μ, σ², hp 
end 


export post_dist_M52I
function post_dist_M52I( x_train, x_test, y_train ) 

    # kernel  
    mZero     = MeanZero() ;            # zero mean function 
    kern      = Mat52Iso( 0.0, 0.0 ) ;        # squared eponential kernel (hyperparams on log scale) 
    log_noise = log(0.1) ;              # (optional) log std dev of obs noise 

    # fit GP 
    gp      = GP(x_train, y_train, mZero, kern, log_noise) 
    optimize!(gp) 
    μ, σ²   = predict_y( gp, x_test )    
    
    # return HPs 
    σ_f = sqrt( gp.kernel.σ2 ) 
    l   = gp.kernel.ℓ   
    σ_n = exp( gp.logNoise.value )  
    hp  = [σ_f, l, σ_n] 

    return μ, σ², hp 
end 


export post_dist_per
function post_dist_per( x_train, x_test, y_train ) 

    # kernel  
    mZero     = MeanZero() ;            # zero mean function 
    kern      = Periodic( 0.0, 0.0, 0.0 ) ;        # squared eponential kernel (hyperparams on log scale) 
    log_noise = log(0.1) ;              # (optional) log std dev of obs noise 

    # fit GP 
    gp      = GP(x_train, y_train, mZero, kern, log_noise) 
    optimize!(gp) 
    μ, σ²   = predict_y( gp, x_test )    
    
    # return HPs 
    σ_f = sqrt( gp.kernel.σ2 ) 
    l   = sqrt( gp.kernel.ℓ2 )   
    σ_n = exp( gp.logNoise.value )  
    hp  = [σ_f, l, σ_n] 

    return μ, σ², hp 
end 
