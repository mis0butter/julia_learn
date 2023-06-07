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
fn          = ode_sine 
plot_option = 1 
t, x, dx_true, dx_fd = ode_states(fn, plot_option) 


## ============================================ ##
# SINDy alone 

λ = 0.1 
n_vars     = size(x, 2) 
poly_order = n_vars 

Ξ_fd   = SINDy_c_recursion(x, dx_fd, 0, λ, poly_order ) 
Ξ_true = SINDy_c_recursion(x, dx_true, 0, λ, poly_order ) 


## ============================================ ##
# split into training and validation data 

train_fraction = 0.7 
t_train, t_test             = split_train_test(t, train_fraction) 
x_train, x_test             = split_train_test(x, train_fraction) 
dx_true_train, dx_true_test = split_train_test(dx_true, train_fraction) 
dx_fd_train, dx_fd_test     = split_train_test(dx_fd, train_fraction) 


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

dx_fn = build_dx_fn(poly_order, z_fd) 
t_val, x_val = validate_data(t_test, x_test, dx_fn)


## ============================================ ##


tspan = (t_test[1], t_test[end])
prob = ODEProblem(dx_fn, x0, tspan)

# solve the ODE
sol = solve(prob,  reltol = 1e-8, abstol = 1e-8)
x_validate = sol.u ; 
x_validate = mapreduce(permutedims, vcat, x_validate) 
t_validate = sol.t 

## ============================================ ##


# display training data 
plot(t_train, x_train, 
    lw = 2, 
    label = "train", 
    grid = false ) 
plot!(t_test, x_test, 
    ls = :dash, 
    c = :red, 
    label = "test" )
plot!(t_val, x_val, 
    ls = :dot, 
    c = :green, 
    label = "validate" )




