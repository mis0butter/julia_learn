using Optim 

## ============================================ ##

export opt_ξ
function opt_ξ( aug_L, ξ, σ_f, l, σ_n, z, u )
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
