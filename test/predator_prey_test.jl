struct Hist 
    objval 
    r_norm 
    s_norm 
    eps_pri 
    eps_dual 
end 

# using DifferentialEquations 
using GaussianSINDy
using LinearAlgebra 
# using ForwardDiff 
# using Optim 
# using Plots 
# using CSV 
# using DataFrames 
# using Symbolics 
# using PrettyTables 
# using Test 
# using NoiseRobustDifferentiation
# using Random, Distributions 

# using LaTeXStrings
# using Latexify


## ============================================ ##
# choose ODE, plot states --> measurements 

#  
fn          = predator_prey 
plot_option = 1 
t, x, dx_true, dx_fd = ode_states(fn, plot_option) 

# split into training and validation data 
train_fraction = 0.7 
t_train, t_test             = split_train_test(t, train_fraction) 
x_train, x_test             = split_train_test(x, train_fraction) 
dx_true_train, dx_true_test = split_train_test(dx_true, train_fraction) 
dx_fd_train, dx_fd_test     = split_train_test(dx_fd, train_fraction) 

## ============================================ ##
# SINDy alone 

λ = 0.1 
n_vars     = size(x, 2) 
poly_order = n_vars 

Ξ_sindy = SINDy_c_recursion(x, dx_fd, 0, λ, poly_order ) 
Ξ_true  = SINDy_c_recursion(x, dx_true, 0, λ, poly_order ) 


## ============================================ ##
# SINDy + GP + ADMM 


λ = 0.02  
println("λ = ", λ)

# finite difference 
hist_fd = Hist( [], [], [], [], [] ) 
@time z_gpsindy, hist_fd = sindy_gp_admm( x_train, dx_fd_train, λ, hist_fd ) 
display(z_gpsindy) 


## ============================================ ##
# generate + validate data 

dx_gpsindy_fn = build_dx_fn(poly_order, z_gpsindy) 
dx_sindy_fn   = build_dx_fn(poly_order, Ξ_sindy)

t_gpsindy_val, x_gpsindy_val = validate_data(t_test, x_test, dx_gpsindy_fn, 0.1) 
t_sindy_val, x_sindy_val     = validate_data(t_test, x_test, dx_sindy_fn, 0.1) 

# plot!! 
plot_prey_predator( t_train, x_train, t_test, x_test, t_sindy_val, x_sindy_val, t_gpsindy_val, x_gpsindy_val ) 

# print some stats 
println("opnorm( Ξ_true - Ξ_sindy ) = \n    ", opnorm( Ξ_true - Ξ_sindy ) )
println("opnorm( Ξ_true - z_fd ) = \n    ", opnorm( Ξ_true - z_gpsindy ) )

# savefig("./sindy_gpsindy.pdf")


