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


## ============================================ ##
# choose ODE, plot states --> measurements 

#  
fn          = lorenz 
plot_option = 1 
t, x, dx_true, dx_fd = ode_states(fn, plot_option) 


## ============================================ ##
# SINDy alone 

λ = 0.1 
poly_order = n_vars 

dx = fdiff(t, x) 
Ξ_fd = SINDy_c_recursion(x, dx, 0, λ, poly_order )

dx = dx_true 
Ξ_true = SINDy_c_recursion(x, dx, 0, λ, poly_order )


## ============================================ ##
# split into training and validation data 

ind = Int(round( size(x, 1) * 0.7 ))  

x_train = x[1:ind,:]
x_test  = x[ind+1:end,:] 

dx_true_train = dx_true[1:ind,:] 
dx_true_test  = dx_true[ind+1:end,:] 

dx_fd_train = dx_fd[1:ind,:] 
dx_fd_test  = dx_fd[ind+1:end] 


## ============================================ ##
# SINDy + GP + ADMM 

# # truth 
# hist_true = Hist( [], [], [], [], [] ) 
# @time z_true, hist_true = sindy_gp_admm( x, dx_true, λ, hist_true ) 
# display(z_true) 

# finite difference 
hist_fd = Hist( [], [], [], [], [] ) 
@time z_fd, hist_fd = sindy_gp_admm( x_train, dx_fd_train, λ, hist_fd ) 
display(z_fd) 



## ============================================ ##
# validation 

# display training data 

display(plt_vec) 


