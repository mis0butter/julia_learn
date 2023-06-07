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

# test 
ξ() = z_fd[:,1] 

# numerically evaluate each function at x and return a vector of numbers
𝚽(x, func_vector) = [f(x) for f in func_vector]

# define functions 
func_vector = pool_data_fn(n_vars, poly_order) 

# define the differential equation
f(x,p,t) = dot( 𝚽(x, func_vector), ξ() ) 

# setup the problem
x0 = 1.0
tspan = (0.0, 1.0)
prob = ODEProblem(f, x0, tspan)

# solve the ODE
sol = solve(prob,  reltol = 1e-8, abstol = 1e-8)

# print the solution
println("Solution at t = 1.0 is: ", sol(1.0))



## ============================================ ##
# validation 

# display training data 

display(plt_vec) 


