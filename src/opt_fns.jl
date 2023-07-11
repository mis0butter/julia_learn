using Optim 
using GaussianProcesses
using LinearAlgebra 

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

## ============================================ ##

export admm_lasso 
function admm_lasso( t, dx, Θx, (ξ, z, u), λ, α, ρ, abstol, reltol, hist ) 
# ----------------------- #
# PURPOSE: 
#       Run one iteration of ADMM LASSO 
# INPUTS: 
#       t           : training data ordinates ( x ) 
#       dx          : training data ( f(x) )
#       Θx          : function library (candidate dynamics) 
#       (ξ, z, u)   : dynamics coefficients primary and dual vars 
#       λ           : L1 norm threshold  
#       α           : relaxation parameter 
#       ρ           : idk what this does, but Boyd sets it to 1 
#       print_vars  : option to display ξ, z, u, hp 
#       abstol      : abs tol 
#       reltol      : rel tol 
#       hist        : diagnostics struct 
# OUTPUTS: 
#       ξ           : output dynamics coefficients (ADMM primary variable x)
#       z           : output dynamics coefficients (ADMM primary variable z)
#       u           : input dual variable 
#       hp          : log-scaled hyperparameters 
#       hist        : diagnostis struct 
# ----------------------- #

    # objective fns 
    f_hp, g, aug_L = obj_fns( dx, Θx, λ, ρ )

    # hp-update (optimization) 
    hp  = opt_hp(t, dx, Θx, ξ) 

    # ξ-update 
    ξ = opt_ξ( aug_L, ξ, z, u, hp ) 
    
    # z-update (soft thresholding) 
    z_old = z 
    ξ_hat = α*ξ + (1 .- α)*z_old 
    z     = shrinkage( ξ_hat + u, λ/ρ ) 

    # u-update 
    u += (ξ_hat - z) 
    
    # push diagnostics 
    n = length(ξ) 
    push!( hist.objval, f_hp(ξ, hp) + g(z) )
    push!( hist.fval, f_hp( ξ, hp ) )
    push!( hist.gval, g(z) ) 
    push!( hist.hp, hp )
    push!( hist.r_norm, norm(ξ - z) )
    push!( hist.s_norm, norm( -ρ*(z - z_old) ) )
    push!( hist.eps_pri, sqrt(n)*abstol + reltol*max(norm(ξ), norm(-z)) ) 
    push!( hist.eps_dual, sqrt(n)*abstol + reltol*norm(ρ*u) ) 
    
    return ξ, z, u, hp, hist 
end 

## ============================================ ##

export gpsindy 
function gpsindy( t, dx_fd, Θx, λ, α, ρ, abstol, reltol ) 
# ----------------------- # 
# PURPOSE: 
#       Main gpsindy function 
# INPUTS: 
#       t       : training data ordinates ( x ) 
#       dx      : training data ( f(x) )
#       Θx      : function library (candidate dynamics) 
#       λ       : L1 norm threshold  
#       α       : relaxation parameter 
#       ρ       : idk what this does, but Boyd sets it to 1 
#       abstol  : abs tol 
#       reltol  : rel tol 
# OUTPUTS: 
#       Ξ       : sparse dynamics coefficient (hopefully) 
#       hist    : diagnostics struct 
# ----------------------- # 

# set up 
hist_nvars = [] 
Ξ          = zeros( size(Θx, 2), size(dx_fd, 2) ) 

# loop with state j
n_vars = size(dx_fd, 2) 
for j = 1 : n_vars

    dx = dx_fd[:,j] 

    # ξ-update 
    n = size(Θx, 2); ξ = z = u = zeros(n) 
    f_hp, g, aug_L = obj_fns( dx, Θx, λ, ρ )
    ξ = opt_ξ( aug_L, ξ, z, u, log.( [1.0, 1.0, 0.1] ) ) 

    hist = Hist( [], [], [], [], [], [], [], [] )  

    # loop until convergence or max iter 
    for k = 1 : 1000  

        # ADMM LASSO! 
        z_old = z 
        ξ, z, u, hp, hist = admm_lasso( t, dx, Θx, (ξ, z, u), λ, α, ρ, abstol, reltol, hist )     

        # end condition 
        if hist.r_norm[end] < hist.eps_pri[end] && hist.s_norm[end] < hist.eps_dual[end] 
            break 
        end 

    end 

    # push diagnostics 
    push!( hist_nvars, hist ) 
    Ξ[:,j] = z 
    
    end 

    return Ξ, hist_nvars  
end 

## ============================================ ##

function monte_carlo_gpsindy( noise_vec )
    
    # choose ODE, plot states --> measurements 
    fn = predator_prey 
    x0, dt, t, x, dx_true, dx_fd = ode_states(fn, 0, 2) 
    
    # SINDy 
    λ = 0.1 ; n_vars = size(x, 2) ; poly_order = n_vars 
    Ξ_true  = SINDy_test( x, dx_true, λ ) 
    
    # function library   
    Θx = pool_data_test(x, n_vars, poly_order) 

    # constants 
    abstol = 1e-2 ; reltol = 1e-2   
    α      = 1.0  ; ρ = 1.0     

    sindy_err_vec   = [] 
    gpsindy_err_vec = [] 
    for dx_noise = noise_vec 
    
        # use this for derivative data noise 
        println( "dx_noise = ", dx_noise )
        dx_fd = dx_true + dx_noise*randn( size(dx_true, 1), size(dx_true, 2) ) 
    
        # SINDy 
        Ξ_sindy = SINDy_test( x, dx_fd, λ ) 
    
        # GPSINDy 
        Ξ_gpsindy, hist_nvars = gpsindy( t, dx_fd, Θx, λ, α, ρ, abstol, reltol )  
    
        sindy_err   = [] 
        gpsindy_err = [] 
        for i = 1:n_vars 
            push!( sindy_err,   norm( Ξ_true[:,i] - Ξ_sindy[:,i] ) )
            push!( gpsindy_err, norm( Ξ_true[:,i] - Ξ_gpsindy[:,i] ) )
        end
        push!( sindy_err_vec,   sindy_err ) 
        push!( gpsindy_err_vec, gpsindy_err ) 
    
    end 

    return sindy_err_vec, gpsindy_err_vec
end 
