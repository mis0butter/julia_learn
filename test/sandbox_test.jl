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

x0, str, p, ts, dt = init_params(fn) 
n_vars = length(x0) 


# ----------------------- #
# solve ODE, plot states 

# solve ODE 
prob = ODEProblem(fn, x0, ts, p) 
sol  = solve(prob, saveat = dt) 

# extract variables --> measurements 
sol_total = sol 
x = sol.u ; x = mapreduce(permutedims, vcat, x) 
t = sol.t 

if plot_option == 1 
    plot_dyn(t, x, str)
end 

# ----------------------- #
# derivatives 
dx_fd   = fdiff(t, x)               # (forward) finite difference 
dx_true = dx_true_fn(t, x, p, fn)   # true derivatives 

    # variational derivatives 
    dx_tv  = 0*x 
    n_vars = size(x, 2)

    for i = 1:n_vars 
        dx = x[2,i] - x[1,i] 
        dx_tv[:,i] = tvdiff(x[:,i], 100, 0.0001)
    end 

    
    
# plot derivatives 
if plot_option == 1 
    plot_deriv(t, dx_true, dx_fd, dx_fd, str) 
end 

## ============================================ ##


n_vars = size(x,2) 
x_train = t 
# x_test  = t 
# x_test  = collect(0 : 0.01 : 10)
x_test = t 

dx_gp = 0 * [x_test x_test]
dx_fd_scale = dx_fd * 100 

for i = 1:n_vars 

    y_train = dx_fd_scale[:,i]

    # f_hp(( σ_f, l, σ_n )) = log_p( σ_f, l, σ_n, x_train, y_train, x_train*0 )

    # # ----------------------- # 
    # # hp-update (optimization) 

    # # bounds 
    # lower = [0.0, 0.0, 0.0]  
    # upper = [Inf, Inf, Inf] 
    # σ_0   = [1.0, 1.0, 0.1]  

    # od     = OnceDifferentiable( f_hp, σ_0 ; autodiff = :forward ) 
    # result = optimize( od, lower, upper, σ_0, Fminbox( LBFGS() ) ) 
        
    # # assign optimized hyperparameters 
    # σ_f = result.minimizer[1] 
    # l   = result.minimizer[2] 
    # σ_n = result.minimizer[3] 

    # μ_post, Σ_post = post_dist( x_train, y_train, x_test, σ_f, l, σ_n )
    # μ_post, Σ_post = post_dist( x_train, y_train, x_test, 1.0, 1.0, 0.1 )

    # covariance from training data 
    K    = k_fn(σ_f, l, x_train, x_train)  
    K   += σ_n^2 * I       # add noise for positive definite 
    Ks   = k_fn(σ_f, l, x_train, x_test)  
    Kss  = k_fn(σ_f, l, x_test, x_test) 
    
    μ_post = Ks' * K^-1 * y_train 
    Σ_post = Kss - (Ks' * K^-1 * Ks)  

    dx_gp[:,i]     = μ_post 

end 

dx_gp = dx_gp / 100 


## ============================================ ##

log_p( 1.0, 1.0, 0.1, x_train, y_train, x_train*0 )
f_hp( σ_f, l, σ_n ) = log_p( σ_f, l, σ_n, x_train, y_train, x_train*0 )
f_hp(1.0, 1.0, 0.1) 



## ============================================ ##
# posterior distribution 

# test data points (PRIOR) 
x_test  = collect( 0 : 0.01 : 2π )
Σ_test  = k_fn( σ_f0, l_0, x_test, x_test )
Σ_test += σ_n0^2 * I 

Kss = k_fn( σ_f0, l_0, x_test, x_test )

# fit data 
# μ_post, Σ_post = post_dist( x_train, y_train, x_test, σ_f0, l_0, σ_n0 )

    # covariance from training data 
    K    = k_fn(σ_f, l, x_train, x_train)  
    K   += σ_n^2 * I       # add noise for positive definite 
    Ks   = k_fn(σ_f, l, x_train, x_test)  
    Kss  = k_fn(σ_f, l, x_test, x_test) 
    
    μ_post = Ks' * K^-1 * y_train 
    Σ_post = Kss - (Ks' * K^-1 * Ks)  

# get covariances and stds 
cov_prior = diag(Kss );     std_prior = sqrt.(cov_prior); 
cov_post  = diag(Σ_post );  std_post  = sqrt.(cov_post); 




