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

# choose ODE, plot states --> measurements 
x0, dt, t, x, dx_true, dx_fd = ode_states(fn, plot_option, fd_method) 

# SINDy 
λ = 0.2 ; n_vars = size(x, 2) ; poly_order = n_vars 
Ξ_true  = SINDy_test( x, dx_true, λ ) 
Ξ_sindy = SINDy_test( x, dx_fd, λ ) 

# function library   
Θx = pool_data_test(x, n_vars, poly_order) 

## ============================================ ##
# GPSINDy 

sindy_err_vec   = [] 
gpsindy_err_vec = [] 
for dx_noise = 0 : 0.1 : 1.0 

    # use this for derivative data noise 
    dx_noise  = 1.0 
    dx_fd = dx_true + dx_noise*randn( size(dx_true, 1), size(dx_true, 2) ) 

    λ = 0.1 ; α = 1.0 ; ρ = 1.0 
    abstol = 1e-2 ; reltol = 1e-2           
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

## ============================================ ##
# back to MONTE CARLO GP SINDy

    Ξ_sindy_err   = [ norm( Ξ_true[:,1] - Ξ_sindy[:,1] ), norm( Ξ_true[:,2] - Ξ_sindy[:,2] )  ] 
    Ξ_gpsindy_err = [ norm( Ξ_true[:,1] - Ξ_gpsindy[:,1] ), norm( Ξ_true[:,2] - Ξ_gpsindy[:,2] )  ] 

    
