using Revise 
using DifferentialEquations 
using GP_june 
using LinearAlgebra 
using Zygote 


## ============================================ ##
# ODE function 
function ODE_test(dx, x, p, t)

    dx[1] = -1/4 * sin(x[1]) ; 
    dx[2] = -1/2 * x[2] ; 

    return dx 

end 

# initial conditions 
x0 = [1; 1] ; 
n_vars = size(x0,1) ; 

# timespan 
ts   = (0.0, 10.0) ; 
dt   = 0.1 ; 

# solve ODE 
prob = ODEProblem(ODE_test, x0, ts) ; 
sol  = solve(prob, saveat = dt) ; 

using Plots 
plot(sol) 


## ============================================ ## 
# finite differencing 

# extract variables 
x = sol.u ; 
t = sol.t ; 

# finite difference 
xdot = 0*x ; 
for i = 1 : length(x)-1
    xdot[i] = ( x[i+1] - x[i] ) / dt ; 
end 
xdot[end] = xdot[end-1] ; 

# true derivatives 
xdot_true = 0*xdot ; 
for i = 1 : length(x) 
    xdot_true[i] = ODE_test([0.0, 0.0], x[i], 0.0, 0.0 ) ; 
end 

println("t type = ", typeof(t)) 
println("xdot_true type = ", typeof(xdot_true)) 
println("xdot type = ", typeof(xdot)) 

# convert vector of vectors into matrix 
xdot_true = mapreduce(permutedims, vcat, xdot_true) 
xdot      = mapreduce(permutedims, vcat, xdot) 

println("t type = ", typeof(t)) 
println("xdot_true type = ", typeof(xdot_true)) 
println("xdot type = ", typeof(xdot)) 

# add to plot 
plot(t, xdot_true, lw = 2) 
plot!(t, xdot, ls = :dot, lw = 2) 
title!("True and Finite Difference Derivatives")
xlabel!("t")
  
  
## ============================================ ##
# TRY FUNCTION 

#define inputs 
n_vars     = 2 ; 
poly_order = n_vars ;

Θ = pool_data(x, n_vars, poly_order) ; 
  

## ============================================ ##
# TRY FUNCTION 

# sparsification knob 
λ = 0.1 ; 

typeof(xdot) 
Ξ = sparsify_dynamics(Θ, xdot, λ, n_vars)


## ============================================ ##
# setting up GP stuff 

# initial hyperparameters 
sig_f0 = 1  
l0     = 1 
sig_n0 = 0.1 

# try first state 
ξ  = XI[:,1] ;   
dx = xdot[:,1] ; 

log_Z(( sig_f, l )) = 1/2*( dx - Θ*ξ )'*inv( sig_f^2 * exp( -1/(2*l^2) * sq_dist(dx,dx) ) )*( dx - Θ*ξ  ) + 1/2*log(det( sig_f^2 * exp( -1/(2*l^2) * sq_dist(dx,dx) ) ))

σ0 = [1.0, 1.0]
result = optimize(log_Z, σ0) 
println("minimizer = ", result.minimizer) 

## ============================================ ##
# ADMM 




