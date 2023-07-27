using Optim 
using GaussianProcesses
using LinearAlgebra 

## ============================================ ##

export l2_metric 
# function l2_metric( n_vars, Θx, Ξ_true, Ξ_sindy, Ξ_gpsindy, sindy_err_vec, gpsindy_err_vec )
function l2_metric( n_vars, dx_train, Θx, Ξ_true, Ξ_sindy, Ξ_gpsindy, sindy_err_vec, gpsindy_err_vec )
# ----------------------- # 
# PURPOSE: 
#       Compute L2 error metric for GPSINDy dynamics coefficients  
# INPUTS: 
#       n_vars 
#       Θx 
#       Ξ_true 
#       Ξ_sindy 
#       Ξ_gpsindy 
#       sindy_err_vec 
#       gpsindy_err_vec 
# OUTPUTS: 
#       sindy_err_vec 
#       gpsindy_err_vec 
# ----------------------- #    
    
    # error metrics  
    sindy_err = [] ; gpsindy_err = [] 
    for i = 1:n_vars 

        # truth derivatives - discovered 
        # push!( sindy_err,   norm( Θx * Ξ_true[:,i] - Θx * Ξ_sindy[:,i] ) )
        # push!( gpsindy_err, norm( Θx * Ξ_true[:,i] - Θx * Ξ_gpsindy[:,i] ) )

        # truth coefficients - discovered 
        push!( sindy_err,   norm( Ξ_true[:,i] - Ξ_sindy[:,i] ) )
        push!( gpsindy_err, norm( Ξ_true[:,i] - Ξ_gpsindy[:,i] ) )

        # training derivatives - discovered 
        # push!( sindy_err,   norm( dx_train[:,i] - Θx * Ξ_sindy[:,i] ) )
        # push!( gpsindy_err, norm( dx_train[:,i] - Θx * Ξ_gpsindy[:,i] ) )
    end
    push!( sindy_err_vec,   sindy_err ) 
    push!( gpsindy_err_vec, gpsindy_err ) 

    return sindy_err_vec, gpsindy_err_vec 

end 

## ============================================ ##

export LU_inv 
function LU_inv( A, b )
# ----------------------- #
# A * x = b --> x = A^-1 * b  
# Use LU factorization instead of matrix inverse 
# ----------------------- #

    C = cholesky(A) 
    L = C.L ; U = C.U 

    y = L \ b 
    x = U \ y 
    
    return x 
end 

## ============================================ ## 

export f_obj 
function f_obj( t, (σ_f, l, σ_n), dx, ξ, x )
# ----------------------- #
# PURPOSE: 
#       Evaluate objective fn 
# INPUTS: 
#       t               : time inputs 
#       (σ_f, l, σ_n)   : hyperparameters 
#       dx              : derivative data inputs 
#       ξ               : dynamics coefficients 
#       x               : state data inputs  
# OUTPUTS: 
#       objval          : objective fn value 
# ----------------------- #

    # kernel is based off of STATES 
    x_vec = [] 
    for i = 1:size(x, 1) 
            x_ind = [] 
            for j = 1:size(x, 2) 
                push!( x_ind, x[i,j] ) 
            end 
        push!( x_vec, [ x[i,1], x[i,2] ] ) 
    end 
    # training kernel function 
    Ky  = k_SE(σ_f, l, x_vec, x_vec) + σ_n^2 * I 
    # Ky  = k_SE(σ_f, l, dx, dx) + (0.1 + σ_n^2) * I 
    # Ky  = k_periodic(σ_f, l, 1.0, dx, dx) + (0.1 + σ_n^2) * I 

    while det(Ky) == 0 
        # println( "det(Ky) = 0" )
        Ky += σ_n * I 
    end 
    
    # # let's say x = inv(Ky)*( dx - Θx*ξ ), or x = inv(A)*b 
    # A       = Ky 
    # b       = ( dx - Θx*ξ ) 
    # x       = LU_inv(A, b) 
    # objval  = 1/2*( dx - Θx*ξ )'*x

    println( "size x = ", size(x) ) 
    n_vars = size(x, 2) ; poly_order = n_vars 

    println( "n_vars = ", n_vars ) 
    println( "poly_order = ", poly_order ) 

    Θx     = pool_data_test( x, n_vars, poly_order ) 

    println( "Θx = ", size(Θx) ) 
    println( "size x = ", size(x) ) 

    y_train = dx - Θx*ξ
    # objval  = 1/2*( y_train )'*inv( Ky )*( y_train ) 
    objval  = 1/2*( y_train )' * ( Ky \ y_train ) 

    println( "size y_train = ", size(y_train) ) 
    println( "size dx = ", size(dx) ) 
    println( "size x = ", size(x) ) 
    println( "size(Ky) = ", size(Ky) ) 

    # scale? 
    # objval += 1/2*sum(log.( Ky )) 
    # objval += 1/2*log( tr(Ky) ) 
    objval += 1/2*log( det(Ky) ) 

    return objval  

end 

## ============================================ ##

export obj_fns 
function obj_fns( t, dx, x, λ, ρ )
# ----------------------- # 
# PURPOSE: 
#       Produce obj fns for SINDy-GP-ADMM 
# INPUTS: 
#       dx      : derivative data 
#       Θx      : function library 
#       λ       : L1 norm threshold 
#       ρ       : idk what this does but Boyd sets it to 1     
# OUTPUTS: 
#       f       : main obj fn  
#       g       : z obj fn 
#       aug_L   : augmented Lagrangian for ADMM 
# ----------------------- # 

    # assign for f_hp_opt 
    f(ξ, hp) = f_obj( t, hp, dx, ξ, x ) 

    # l1 norm 
    g(z) = λ * sum(abs.(z)) 

    # augmented Lagrangian (scaled form) 
    aug_L(ξ, hp, z, u) = f(ξ, hp) + g(z) + ρ/2 .* norm( ξ - z + u )^2      

    return f, g, aug_L 
end 

## ============================================ ##

export opt_ξ
function opt_ξ( aug_L, ξ, z, u, hp )
# ----------------------- #
# PURPOSE: 
#       Optimize ξ for augmented Lagrangian obj fn 
# INPUTS: 
#       aug_L   : augmented Lagrangian for SINDy-GP-ADMM 
#       ξ       : input dynamics coefficients (ADMM primary variable x)
#       z       : input dynamics coefficients (ADMM primary variable z)
#       u       : dual variable 
#       hp      : hyperparameters 
# OUTPUTS: 
#       ξ       : output dynamics coefficient (ADMM primary variable x) 
# ----------------------- #

    # ξ-update 

    # optimization 
    f_opt(ξ) = aug_L(ξ, hp, z, u) 

    println( "f_opt(0) = ", f_opt( zeros(8) ) )

    od       = OnceDifferentiable( f_opt, ξ ; autodiff = :forward ) 
    result   = optimize( od, ξ, LBFGS() ) 
    ξ        = result.minimizer 

    return ξ 
end 

## ============================================ ##

export opt_hp 
function opt_hp(t_train, dx_train, Θx, ξ) 
# ----------------------- #
# PURPPOSE: 
#       Optimize hyperparameters for marginal likelihood of data  
# INPUTS: 
#       t_train     : training data ordinates ( x ) 
#       dx_train    : training data ( f(x) )
#       Θx          : function library (candidate dynamics) 
#       ξ           : dynamics coefficients 
# OUTPUTS: 
#       hp          : log-scaled hyperparameters 
# ----------------------- #

    # kernel  
    mZero     = MeanZero() ;            # zero mean function 
    kern      = SE( 0.0, 0.0 ) ;        # squared eponential kernel (hyperparams on log scale) 
    log_noise = log(0.1) ;              # (optional) log std dev of obs noise 

    # fit GP 
    # y_train = dx_train 
    y_train = dx_train - Θx*ξ   
    # y_train = Θx*ξ
    gp      = GP(t_train, y_train, mZero, kern, log_noise) 
    # gp  = GP(t_train, dx_train, Θx*ξ, kern, log_noise) 

    # ----------------------- #
    # optimize 

    optimize!(gp) 

    σ_f = sqrt( gp.kernel.σ2 ) 
    l   = sqrt( gp.kernel.ℓ2 )  
    σ_n = exp( gp.logNoise.value )  
    hp  = [σ_f, l, σ_n] 

    # check for 0 
    for i = 1:3 
        if abs(hp[i]) < 1e-10 
            hp[i] = 1e-10 
        end 
    end 

    return hp 
end 

## ============================================ ##

export shrinkage 
function shrinkage(x, κ) 
# ----------------------- #
# shrinkage / L1-norm min / soft thresholding 
# ----------------------- #

    z = 0*x ; 
    for i = 1:length(x) 
        z[i] = max( 0, x[i] - κ ) - max( 0, -x[i] - κ ) 
    end 

    return z 
end 

