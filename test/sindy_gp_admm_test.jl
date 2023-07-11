
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

fn             = predator_prey 
plot_option    = 0 
savefig_option = 0 
fd_method      = 2 # 1 = forward, 2 = central, 3 = backward 
abstol         = 1e-2 
reltol         = 1e-2           

# choose ODE, plot states --> measurements 
x0, dt, t, x, dx_true, dx_fd = ode_states(fn, plot_option, fd_method) 

# SINDy 
λ          = 0.2 
n_vars     = size(x, 2) 
poly_order = n_vars 

Ξ_true  = SINDy_test( x, dx_true, λ ) 
Ξ_sindy = SINDy_test( x, dx_fd, λ ) 

# function library   
Θx = pool_data_test(x, n_vars, poly_order) 

dx_noise  = 1.0 

# ----------------------- #
# MONTE CARLO GPSINDY 

    dx_fd = dx_true + dx_noise*randn( size(dx_true, 1), size(dx_true, 2) ) 

## ============================================ ##
## ============================================ ##
z_nvars, hist_nvars = gpsindy( t, dx_fd, Θx, 0.1, α, ρ, abstol, reltol )  

## ============================================ ##
# sindy_gp_admm 

    # ----------------------- #
    
    hist_nvars = [] 

    # loop with state j
    n_vars = size(dx_fd, 2) 
    for j = 1 : n_vars

        dx = dx_fd[:,j] 

        z, hist = gpsindy( t, dx, Θx, λ, α, ρ, abstol, reltol )  
        push!( hist_nvars, hist ) 

    end 

## ============================================ ##
# back to MONTE CARLO GP SINDy

    Ξ_sindy_err   = [ norm( Ξ_true[:,1] - Ξ_sindy[:,1] ), norm( Ξ_true[:,2] - Ξ_sindy[:,2] )  ] 
    z_gpsindy_err = [ norm( Ξ_true[:,1] - z_gpsindy[:,1] ), norm( Ξ_true[:,2] - z_gpsindy[:,2] )  ] 

    
