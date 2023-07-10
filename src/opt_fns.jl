using Optim 
using GaussianProcesses

## ============================================ ##

export opt_ξ
function opt_ξ( aug_L, ξ, σ_f, l, σ_n, z, u )
# PURPOSE: 
#       Optimize ξ for augmented Lagrangian obj fn 
# INPUTS: 
#       aug_L   : augmented Lagrangian for SINDy-GP-ADMM 
#       ξ       : input dynamics coefficients (ADMM primary variable x)
#       σ_f     : signal noise hyperparameter 
#       l       : length scale hyperparameter 
#       σ_n     : observation noise hyperparameter 
#       z       : : input dynamics coefficients (ADMM primary variable z)
#       u       : dual variable 
# OUTPUTS: 
#       ξ       : output dynamics coefficient (ADMM primary variable x) 

    # optimization 
    f_opt(ξ) = aug_L(ξ, exp(σ_f), exp(l), exp(σ_n), z, u) 
    od       = OnceDifferentiable( f_opt, ξ ; autodiff = :forward ) 
    result   = optimize( od, ξ, LBFGS() ) 
    ξ        = result.minimizer 

    return ξ
end 

## ============================================ ##

export opt_hp 
function opt_hp(t_train, dx_train, Θx, ξ) 
# PURPPOSE: 
#       Optimize hyperparameters for marginal likelihood of data  
# INPUTS: 
#       t           : training data ordinates ( x ) 
#       dx_train    : training data ( f(x) )
#       Θx          : function library (candidate dynamics) 
#       ξ           : dynamics coefficients 
# OUTPUTS: 
#       hp          : hyperparameters 

    # mean and covariance 
    mZero = MeanZero() ;            # zero mean function 
    kern  = SE( 0.0, 0.0 ) ;          # squared eponential kernel (hyperparams on log scale) 
    log_noise = log(0.1) ;              # (optional) log std dev of obs noise 

    # fit GP 
    # y_train = dx_train - Θx*ξ   
    y_train = Θx*ξ
    gp  = GP(t_train, y_train, mZero, kern, log_noise) 
    # gp  = GP(t_train, dx_train, Θx*ξ, kern, log_noise) 
    result = optimize!(gp) 

    σ_f = result.minimizer[1] 
    l   = result.minimizer[2] 
    σ_n = result.minimizer[3] 
    hp  = [σ_f, l, σ_n] 

    return hp 
end 
