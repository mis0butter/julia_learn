struct Hist 
    objval 
    r_norm 
    s_norm 
    eps_pri 
    eps_dual 
end 

using DifferentialEquations 
using GaussianSINDy
using LinearAlgebra 
using ForwardDiff 
using Optim 
using Plots 
using CSV 
using DataFrames 
using Symbolics 
using PrettyTables 
using Test 
using NoiseRobustDifferentiation
using Random, Distributions 


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

Ξ_fd   = SINDy_c_recursion(x, dx_fd, 0, λ, poly_order ) 
Ξ_true = SINDy_c_recursion(x, dx_true, 0, λ, poly_order ) 


## ============================================ ##
# SINDy + GP + ADMM 

# # truth 
# hist_true = Hist( [], [], [], [], [] ) 
# @time z_true, hist_true = sindy_gp_admm( x, dx_true, λ, hist_true ) 
# display(z_true) 

λ = 0.01 

# finite difference 
hist_fd = Hist( [], [], [], [], [] ) 
@time z_fd, hist_fd = sindy_gp_admm( x_train, dx_true_train, λ, hist_fd ) 
display(z_fd) 


## ============================================ ##
# generate + validate data 

dx_gpsindy_fn = build_dx_fn(poly_order, z_fd) 
dx_sindy_fn  = build_dx_fn(poly_order, Ξ_fd)

t_gpsindy_val, x_gpsindy_val = validate_data(t_test, x_test, dx_gpsindy_fn)
t_sindy_val, x_sindy_val     = validate_data(t_test, x_test, dx_sindy_fn)

# display training data 
plot(t_train, x_train, 
    lw = 3, 
    c = :black, 
    label = "train", 
    grid = false, 
    xlim = (t_train[end]*3/4, t_test[end]), 
    legend 
    # ylim = ( min(x_test[1], x_test[end]), max( x_test[1], x_test[end] ) )
    ) 
plot!(t_test, x_test, 
    ls = :dash, 
    c = :blue,
    lw = 3,  
    label = "test" 
    )
plot!(t_gpsindy_val, x_gpsindy_val, 
    ls = :dashdot, 
    lw = 1.5, 
    c = :red, 
    label = "GP SINDy" 
    )
plot!(t_sindy_val, x_sindy_val, 
    ls = :dot, 
    lw = 1.5, 
    c = :green, 
    label = "SINDy" 
    )

## ============================================ ##



