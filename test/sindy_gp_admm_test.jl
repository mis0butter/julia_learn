
struct Hist 
    objval 
    fval 
    gval 
    hp 
    r_norm 
    s_norm 
    eps_pri 
    eps_dual 
end 

using GaussianSINDy
using LinearAlgebra 
using Plots 
using Dates 
using Optim 
using GaussianProcesses

## ============================================ ##
# choose ODE, plot states --> measurements 

#  
fn             = predator_prey 
plot_option    = 0 
savefig_option = 0 
fd_method      = 2 # 1 = forward, 2 = central, 3 = backward 

# choose ODE, plot states --> measurements 
x0, dt, t, x, dx_true, dx_fd = ode_states(fn, plot_option, fd_method) 

# truth coeffs 
Ξ_true = SINDy_test( x, dx_true, 0.1 ) 
Ξ_true = Ξ_true[:,1] 
 
dx_noise  = 1.0 

# ----------------------- #
# MONTE CARLO GPSINDY 

    dx_fd = dx_true + dx_noise*randn( size(dx_true, 1), size(dx_true, 2) ) 

## ============================================ ##
## ============================================ ##
## ============================================ ##
# sindy_gp_admm 

    # ----------------------- #
    # SINDy 
    
    λ          = 0.2 
    n_vars     = size(x, 2) 
    poly_order = n_vars 

    Ξ_true  = SINDy_test( x, dx_true, λ ) 
    Ξ_sindy = SINDy_test( x, dx_fd, λ ) 

    # SINDy  
    Θx = pool_data_test(x, n_vars, poly_order) 
    Ξ  = sparsify_dynamics_test(Θx, dx_fd, λ, n_vars) 

    n = size(Ξ, 1)

    # ----------------------- #
    # loop with state j
    
    hist_nvars = [] 

    j = 1 
    println( "j = ", j )
    # for j = 1 : n_vars

    # initial loss function vars 
    dx = dx_fd[:,j] 

    abstol   = 1e-2 
    reltol   = 1e-2           # save matrix-vector multiply 

## ============================================ ##
z, hist = gpsindy( t, dx, Θx, λ, α, ρ, abstol, reltol )     
## ============================================ ##

    hist = Hist( [], [], [], [], [], [], [], [] )  

    # ξ-update (optimization) 
    ξ = z = u = zeros(n) 
    f_hp, g, aug_L = obj_fns( dx, Θx, λ, ρ )
    ξ  = opt_ξ( aug_L, ξ, z, u, log.( [ 1.0, 1.0, 0.1 ] ) ) 

    # define constants 
    max_iter = 1000  
    α = 1.0 ; ρ = 1.0 

    iter = 0 
    for k = 1:max_iter 

        # increment counter 
        iter += 1 
        println( "iter = ", iter )

        # ADMM LASSO! 
        z_old = z 
        ξ, z, u, hp, hist = admm_lasso( t, dx, Θx, (ξ, z, u), λ, α, ρ, abstol, reltol, hist )     

        # push diagnostics 
        println( "ξ = ", ξ ) ; println( "z = ", z ) ; println( "hp = ", hp )

        if hist.r_norm[end] < hist.eps_pri[end] && hist.s_norm[end] < hist.eps_dual[end] 
            println("converged!")  
            println( "gpsindy err = ", norm( Ξ_true[:,j] - z ) ) 
            println( "sindy err   = ", norm( Ξ_true[:,j] - Ξ_sindy[:,j] ) ) 
            push!(hist_nvars, hist)
            break 
        end 

    end 
    
## ============================================ ##
# back to MONTE CARLO GP SINDy

    Ξ_sindy_err   = [ norm( Ξ_true[:,1] - Ξ_sindy[:,1] ), norm( Ξ_true[:,2] - Ξ_sindy[:,2] )  ] 
    z_gpsindy_err = [ norm( Ξ_true[:,1] - z_gpsindy[:,1] ), norm( Ξ_true[:,2] - z_gpsindy[:,2] )  ] 

    
