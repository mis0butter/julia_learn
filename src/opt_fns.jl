using Optim 
using GaussianProcesses
using LinearAlgebra 

## ============================================ ##

export l2_metric 
function l2_metric( n_vars, Ξ_true, Ξ_sindy, Ξ_gpsindy, sindy_err_vec, gpsindy_err_vec )
# ----------------------- # 
# PURPOSE: 
#       Compute L2 error metric for GPSINDy dynamics coefficients  
# INPUTS: 
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
        push!( sindy_err,   norm( Ξ_true[:,i] - Ξ_sindy[:,i] ) )
        push!( gpsindy_err, norm( Ξ_true[:,i] - Ξ_gpsindy[:,i] ) )
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

    C = cholesky(A) ; 
    L = C.L ; U = C.U 

    y = L \ b 
    x = U \ y 
    
    return x 
end 

## ============================================ ## 

export f_obj 
function f_obj( (σ_f, l, σ_n), dx, ξ, Θx )
# ----------------------- #
# PURPOSE: 
#       Evaluate objective fn 
# INPUTS: 
#       (σ_f, l, σ_n)   : hyperparameters 
#       dx              : derivative data inputs 
#       ξ               : dynamics coefficients 
#       Θx              : function library of dynamics 
# OUTPUTS: 
#       objval          : objective fn value 
# ----------------------- #

    # training kernel function 
    Ky  = k_SE(σ_f, l, dx, dx) + σ_n^2 * I 
    # Ky  = k_SE(σ_f, l, dx, dx) + (0.1 + σ_n^2) * I 
    # Ky  = k_periodic(σ_f, l, 1.0, dx, dx) + (0.1 + σ_n^2) * I 

    while det(Ky) == 0 
        println( "det(Ky) = 0" )
        Ky += σ_n * I 
    end 
    
    # let's say x = inv(Ky)*( dx - Θx*ξ ), or x = inv(A)*b 
    A       = Ky 
    b       = ( dx - Θx*ξ ) 
    x       = LU_inv(A, b) 
    objval  = 1/2*( dx - Θx*ξ )'*x

    # objval  = 1/2*( dx - Θx*ξ )'*inv( Ky )*( dx - Θx*ξ ) 

    # scale? 
    objval += 1/2*log( tr(Ky) ) 

    return objval  

end 

## ============================================ ##

export obj_fns 
function obj_fns( dx, Θx, λ, ρ )
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
    f(ξ, hp) = f_obj( hp, dx, ξ, Θx )

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
#       hp      : log-scaled hyperparameters 
# OUTPUTS: 
#       ξ       : output dynamics coefficient (ADMM primary variable x) 
# ----------------------- #

    # optimization 
    f_opt(ξ) = aug_L(ξ, exp.(hp), z, u) 
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
    # y_train = dx_train - Θx*ξ   
    y_train = Θx*ξ
    gp      = GP(t_train, y_train, mZero, kern, log_noise) 
    # gp  = GP(t_train, dx_train, Θx*ξ, kern, log_noise) 
    result  = optimize!(gp) 

    σ_f = result.minimizer[1] 
    l   = result.minimizer[2] 
    σ_n = result.minimizer[3] 
    hp  = [σ_f, l, σ_n] 

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

