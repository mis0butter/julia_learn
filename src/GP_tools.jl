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
            C[i,j] = norm( a[i] - b[j] )^2 
        end 
    end 

    return C 

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

    # term  = 1/2 * ( y - μ )' * inv( Ky ) * ( y - μ ) 
    term  = 1/2 * ( y - μ )' * ( Ky \ ( y - μ ) ) 
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
# posterior GP and optimize hps 

export gp_post 
function gp_post( x_test, μ_prior, x_train, μ_train, y_train ) 
# ----------------------- #
# PURPOSE: 
#       Compute posterior of Gaussian process and optimize hyperparameters 
# INPUTS: 
#       x_test  : TEST input points for the PRIOR model 
#       μ_prior : mean function m(x) of the PRIOR model 
#       x_train : input points for the TRAINING data 
#       y_train : output points for the TRAINING data 
#       μ_train : mean function m(x) for the TRAINING data 
# OUTPUTS: 
#       y_post  : 
# ----------------------- # 

    # transform x inputs matrices --> vectors for kernel computations 
    x_test_vec = [] 
    for i = 1 : size(x_test, 1) 
        push!( x_test_vec, x_test[i,:] ) 
    end 
    x_train_vec = [] 
    for i = 1 : size(x_train, 1) 
        push!( x_train_vec, x_train[i,:] ) 
    end 

    # set up posterior 
    r      = size(x_test, 1) ; n_vars = size(y_train, 2) 
    y_post = zeros( r, n_vars ) 
    
    # optimize hyperparameters, compute posterior y_post for each state 
    for i = 1 : n_vars 
    
        # kernel  
        mZero     = MeanZero()              # zero mean function 
        kern      = SE( 0.0, 0.0 )          # squared eponential kernel (hyperparams on log scale) 
        log_noise = log(0.1)                # (optional) log std dev of obs 
        
        # y_train = dx_noise[:,i] - dx_mean[:,i]
        gp      = GP( x_train', y_train[:,i] - μ_train[:,i], mZero, kern, log_noise ) 
        optimize!( gp, method = LBFGS( linesearch = LineSearches.BackTracking() ) ) 
    
        # return HPs 
        σ_f = sqrt( gp.kernel.σ2 ) ; l = sqrt.( gp.kernel.ℓ2 ) ; σ_n = exp( gp.logNoise.value )  
        hp  = [σ_f, l, σ_n] 
    
        # compute kernels 
        Kss = k_SE( σ_f, l, x_test_vec, x_test_vec ) 
        Ks  = k_SE( σ_f, l, x_test_vec, x_train_vec ) 
        K   = k_SE( σ_f, l, x_train_vec, x_train_vec ) 

        # posterior 
        y_post[:,i] = μ_prior[:,i] + Ks * ( ( K + σ_n^2 * I ) \ ( y_train[:,i] - μ_train[:,i] ) ) 
        Σ           = Kss - Ks * ( ( K + σ_n^2 * I ) \ Ks' ) 
    
    end 

    return y_post 

end 


## ============================================ ##
# hp optimization (June) --> post mean  

export post_dist_hp_opt 
function post_dist_hp_opt( x_train, y_train, y_mean, x_test, plot_option = false )

    # IC 
    hp = [ 1.0, 1.0, 0.1 ] 

    # optimization 
    hp_opt(( σ_f, l, σ_n )) = log_p( σ_f, l, σ_n, x_train, y_train, y_mean )
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
using Optim 
using LineSearches 

export post_dist_SE 
function post_dist_SE( x_train, y_train, x_test ) 

    # kernel  
    mZero     = MeanZero() ;            # zero mean function 
    kern      = SE( 0.0, 0.0 ) ;        # squared eponential kernel (hyperparams on log scale) 
    log_noise = log(0.1) ;              # (optional) log std dev of obs noise 
    
    n_vars   = size(y_train, 2) 
    y_smooth = zeros( size(x_test, 1), n_vars ) 
    Σ        = 0 * y_smooth 
    hps      = [] 
    for i = 1:n_vars 

        # fit GP 
        gp      = GP( x_train', y_train[:,i], mZero, kern, log_noise ) 

        # optimize!( gp, method = LBFGS(linesearch=LineSearches.BackTracking()) ) 
        optimize!( gp, method = BFGS(linesearch=LineSearches.BackTracking()) ) 

        # return HPs 
        σ_f = sqrt( gp.kernel.σ2 ) ; l = sqrt.( gp.kernel.ℓ2 ) ; σ_n = exp( gp.logNoise.value )  
        hp  = [σ_f, l, σ_n] 

        μ, σ²   = predict_y( gp, x_test' )  

        y_smooth[:,i] = μ 
        Σ[:,i]        = σ²
        push!( hps, hp ) 
    
    end 

    if n_vars == 1 
        hps = hps[1] 
    end 

    return y_smooth, Σ, hps      
end 

export post_dist_M12A
function post_dist_M12A( x_train, y_train, x_test ) 

    # kernel  
    mZero     = MeanZero() ;            # zero mean function 
    kern      = Mat12Ard( [0.0], 0.0 ) ;        # squared eponential kernel (hyperparams on log scale) 
    log_noise = log(0.1) ;              # (optional) log std dev of obs noise 

    n_vars   = size(y_train, 2) 
    y_smooth = zeros( length(x_test), n_vars ) 
    Σ        = 0 * y_smooth 
    hps      = [] 
    for i = 1:n_vars 

        # fit GP 
        gp      = GP(x_train, y_train[:,i], mZero, kern, log_noise) 
        optimize!(gp) 
        μ, σ²   = predict_y( gp, x_test )  
        
        # return HPs 
        σ_f = sqrt( gp.kernel.σ2 ) ; l = sqrt.( gp.kernel.iℓ2[1] ) ; σ_n = exp( gp.logNoise.value )  
        hp  = [σ_f, l, σ_n] 

        y_smooth[:,i] = μ 
        Σ[:,i]        = σ²
        push!( hps, hp ) 
    
    end 

    return y_smooth, Σ, hps  
end 


export post_dist_M32A
function post_dist_M32A( x_train, y_train, x_test ) 

    # kernel  
    mZero     = MeanZero() ;            # zero mean function 
    kern      = Mat32Ard( [0.0], 0.0 ) ;        # squared eponential kernel (hyperparams on log scale) 
    log_noise = log(0.1) ;              # (optional) log std dev of obs noise 

    n_vars   = size(y_train, 2) 
    y_smooth = zeros( length(x_test), n_vars ) 
    Σ        = 0 * y_smooth 
    hps      = [] 
    for i = 1:n_vars 

        # fit GP 
        gp      = GP(x_train, y_train[:,i], mZero, kern, log_noise) 
        optimize!(gp) 
        μ, σ²   = predict_y( gp, x_test )  
        
        # return HPs 
        σ_f = sqrt( gp.kernel.σ2 ) ; l = sqrt.( gp.kernel.iℓ2[1] ) ; σ_n = exp( gp.logNoise.value )  
        hp  = [σ_f, l, σ_n] 
        
        y_smooth[:,i] = μ 
        Σ[:,i]        = σ²
        push!( hps, hp ) 
    
    end 

    return y_smooth, Σ, hps  
end 


export post_dist_M52A
function post_dist_M52A( x_train, y_train, x_test ) 

    # kernel  
    mZero     = MeanZero() ;            # zero mean function 
    kern      = Mat52Ard( [0.0], 0.0 ) ;        # squared eponential kernel (hyperparams on log scale) 
    log_noise = log(0.1) ;              # (optional) log std dev of obs noise 

    n_vars   = size(y_train, 2) 
    y_smooth = zeros( length(x_test), n_vars ) 
    Σ        = 0 * y_smooth 
    hps      = [] 
    for i = 1:n_vars 

        # fit GP 
        gp      = GP(x_train, y_train[:,i], mZero, kern, log_noise) 
        optimize!(gp) 
        μ, σ²   = predict_y( gp, x_test )  
        
        # return HPs 
        σ_f = sqrt( gp.kernel.σ2 ) ; l = sqrt.( gp.kernel.iℓ2[1] ) ; σ_n = exp( gp.logNoise.value )  
        hp  = [σ_f, l, σ_n] 

        y_smooth[:,i] = μ 
        Σ[:,i]        = σ²
        push!( hps, hp ) 
    
    end 

    return y_smooth, Σ, hps  
end 


export post_dist_M12I
function post_dist_M12I( x_train, y_train, x_test ) 

    # kernel  
    mZero     = MeanZero() ;            # zero mean function 
    kern      = Mat12Iso( 0.0, 0.0 ) ;        # squared eponential kernel (hyperparams on log scale) 
    log_noise = log(0.1) ;              # (optional) log std dev of obs noise 
    
    n_vars   = size(y_train, 2) 
    y_smooth = zeros( length(x_test), n_vars ) 
    Σ        = 0 * y_smooth 
    hps      = [] 
    for i = 1:n_vars 

        # fit GP 
        gp      = GP(x_train, y_train[:,i], mZero, kern, log_noise) 
        optimize!(gp) 
        μ, σ²   = predict_y( gp, x_test )  
        
        # return HPs 
        σ_f = sqrt( gp.kernel.σ2 ) ; l = gp.kernel.ℓ ; σ_n = exp( gp.logNoise.value )  
        hp  = [σ_f, l, σ_n] 

        y_smooth[:,i] = μ 
        Σ[:,i]        = σ²
        push!( hps, hp ) 
    
    end 

    return y_smooth, Σ, hps 
end 


export post_dist_M32I
function post_dist_M32I( x_train, y_train, x_test ) 

    # kernel  
    mZero     = MeanZero() ;            # zero mean function 
    kern      = Mat32Iso( 0.0, 0.0 ) ;        # squared eponential kernel (hyperparams on log scale) 
    log_noise = log(0.1) ;              # (optional) log std dev of obs noise 

    n_vars   = size(y_train, 2) 
    y_smooth = zeros( length(x_test), n_vars ) 
    Σ        = 0 * y_smooth 
    hps      = [] 
    for i = 1:n_vars 

        # fit GP 
        gp      = GP(x_train, y_train[:,i], mZero, kern, log_noise) 
        optimize!(gp) 
        μ, σ²   = predict_y( gp, x_test )  
        
        # return HPs 
        σ_f = sqrt( gp.kernel.σ2 ) ; l = gp.kernel.ℓ ; σ_n = exp( gp.logNoise.value )  
        hp  = [σ_f, l, σ_n]   
    
        y_smooth[:,i] = μ 
        Σ[:,i]        = σ²
        push!( hps, hp ) 
    
    end 

    return y_smooth, Σ, hps 
end 


export post_dist_M52I
function post_dist_M52I( x_train, y_train, x_test ) 

    # kernel  
    mZero     = MeanZero() ;            # zero mean function 
    kern      = Mat52Iso( 0.0, 0.0 ) ;        # squared eponential kernel (hyperparams on log scale) 
    log_noise = log(0.1) ;              # (optional) log std dev of obs noise 

    n_vars   = size(y_train, 2) 
    y_smooth = zeros( length(x_test), n_vars ) 
    Σ        = 0 * y_smooth 
    hps      = [] 
    for i = 1:n_vars 

        # fit GP 
        gp      = GP(x_train, y_train[:,i], mZero, kern, log_noise) 
        optimize!(gp) 
        μ, σ²   = predict_y( gp, x_test )  
        
        # return HPs 
        σ_f = sqrt( gp.kernel.σ2 ) ; l = gp.kernel.ℓ ; σ_n = exp( gp.logNoise.value )  
        hp  = [σ_f, l, σ_n]   
    
        y_smooth[:,i] = μ 
        Σ[:,i]        = σ²
        push!( hps, hp ) 
    
    end 

    return y_smooth, Σ, hps 
end 


export post_dist_per
function post_dist_per( x_train, y_train, x_test ) 

    # kernel  
    mZero     = MeanZero() ;            # zero mean function 
    kern      = Periodic( 0.0, 0.0, 0.0 ) ;        # squared eponential kernel (hyperparams on log scale) 
    log_noise = log(0.1) ;              # (optional) log std dev of obs noise 

    n_vars   = size(y_train, 2) 
    y_smooth = zeros( length(x_test), n_vars ) 
    Σ        = 0 * y_smooth 
    hps      = [] 
    for i = 1:n_vars 

        # fit GP 
        gp      = GP(x_train, y_train[:,i], mZero, kern, log_noise) 
        optimize!(gp) 
        μ, σ²   = predict_y( gp, x_test )  
        
        # return HPs 
        σ_f = sqrt( gp.kernel.σ2 ) ; l = sqrt( gp.kernel.ℓ2 ) ; σ_n = exp( gp.logNoise.value )  
        hp  = [σ_f, l, σ_n] 
    
        y_smooth[:,i] = μ 
        Σ[:,i]        = σ²
        push!( hps, hp ) 
    
    end 

    return y_smooth, Σ, hps 
end 
