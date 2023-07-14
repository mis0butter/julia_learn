using Optim 
using GaussianProcesses
using LinearAlgebra 
using Statistics 

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
#       Main gpsindy function (iterate j = 1 : n_vars) 
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
Ξ = zeros( size(Θx, 2), size(dx_fd, 2) ) 
hist_nvars = [] 

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

export monte_carlo_gpsindy 
function monte_carlo_gpsindy( noise_vec, λ, abstol, reltol, case ) 
# ----------------------- #
# PURPOSE:  
#       Run GPSINDy monte carlo 
# INPUTS: 
#       noise_vec       : dx noise vector for iterations 
#       λ               : L1 norm threshold 
#       abstol          : abs tol 
#       reltol          : rel tol 
#       case            : 0 = true, 1 = noise, 2 = norm 
# OUTPUTS: 
#       sindy_err_vec   : sindy error stats 
#       gpsindy_err_vec : gpsindy error stats 
# ----------------------- # 
    
    # choose ODE, plot states --> measurements 
    fn = predator_prey 
    x0, dt, t, x_true, dx_true, dx_fd = ode_states(fn, 0, 2) 
    
    # truth coeffs 
    n_vars = size(x_true, 2) ; poly_order = n_vars 
    Ξ_true = SINDy_test( x_true, dx_true, λ ) 

    # constants 
    α = 1.0  ; ρ = 1.0     

    sindy_err_vec = [] ; gpsindy_err_vec = [] ; hist_nvars_vec = [] 
    for noise = noise_vec 
    
        
        # use true data 
        if case == 0 
            
            Ξ_sindy = SINDy_test( x_true, dx_true, λ ) 
            Θx      = pool_data_test(x_true, n_vars, poly_order) 
            Ξ_gpsindy, hist_nvars = gpsindy( t, dx_true, Θx, λ, α, ρ, abstol, reltol )  

        # use noisy data  
        elseif case == 1 

            # add noise 
            println( "noise = ", noise ) 

            x_noise  = x_true + 0 * noise*randn( size(x_true, 1), size(x_true, 2) )
            # dx_noise = fdiff(t, x_noise, 2)
            dx_noise = dx_true + noise*randn( size(dx_true, 1), size(dx_true, 2) )

            Ξ_sindy = SINDy_test( x_noise, dx_noise, λ ) 
            Θx      = pool_data_test(x_noise, n_vars, poly_order) 
            Ξ_gpsindy, hist_nvars = gpsindy( t, dx_noise, Θx, λ, α, ρ, abstol, reltol )  
    
        # use normalized (true) data 
        else 

            x_norm  = norm_data( t, x_true) 
            # dx_norm = fdiff(t, x_norm, 2) 
            dx_norm = norm_data( t, dx_true )

            Ξ_true  = SINDy_test( x_norm, dx_norm, λ ) 
            Ξ_sindy = SINDy_test( x_norm, dx_norm, λ ) 
            Θx      = pool_data_test(x_norm, n_vars, poly_order) 
            Ξ_gpsindy, hist_nvars = gpsindy( t, dx_norm, Θx, λ, α, ρ, abstol, reltol )  

        end 

        # metrics & diagnostics 
        push!( hist_nvars_vec, hist_nvars )
        sindy_err_vec, gpsindy_err_vec = l2_metric( n_vars, Ξ_true, Ξ_sindy, Ξ_gpsindy, sindy_err_vec, gpsindy_err_vec )
    
    end 

    # make matrices 
    sindy_err_vec   = mapreduce(permutedims, vcat, sindy_err_vec)
    gpsindy_err_vec = mapreduce(permutedims, vcat, gpsindy_err_vec)

    return sindy_err_vec, gpsindy_err_vec, hist_nvars_vec 
end 
