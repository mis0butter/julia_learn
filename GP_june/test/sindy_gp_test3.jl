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

# log_Z function 
log_Z(( sig_f, l, sig_n )) = 1/2*( dx - Θ*ξ )'*inv( sig_f^2 * exp( -1/(2*l^2) * sq_dist(dx,dx) )  + sig_n^2*I )*( dx - Θ*ξ  ) + 1/2*log(det( sig_f^2 * exp( -1/(2*l^2) * sq_dist(dx,dx) ) ))

σ0 = [1.0, 1.0, 0.1]
result = optimize(log_Z, σ0) 
println("minimizer = ", result.minimizer) 

## ============================================ ##
# full augmented Lagrangian 

ρ = 1.0 
λ = 0.1 
y = 0*ξ 
z = 0*ξ
α = 1.0 

function aug_L(( sig_f, l, sig_n, dx, ξ, Θ, y, z, λ, ρ ))

    term = zeros(5) 
    term[1] = 1/2*( dx - Θ*ξ )'*inv( sig_f^2 * exp( -1/(2*l^2) * sq_dist(dx,dx) ) + sig_n^2*I )*( dx - Θ*ξ  ) 
    term[2] = 1/2*log(det( sig_f^2 * exp( -1/(2*l^2) * sq_dist(dx,dx) ) )) 
    term[3] = λ*sum(abs.(z)) 
    term[4] = y'*(ξ-z) 
    term[5] = ρ/2*( norm(ξ-z) )^2 

    return sum(term)

end 

# test 
aug_L(( sig_f0, l0, sig_n0, dx, ξ, Θ, y, z, λ, ρ ))

aug_L_hp(( sig_f, l, sig_n )) = aug_L(( sig_f, l, sig_n, dx, ξ, Θ, y, z, λ, ρ ))
aug_L_hp((sig_f0, l0, sig_n0))

result = optimize(aug_L_hp, σ0)
println("aug_L_hp min = ", result.minimizer) 


## ============================================ ##
# ADMM 

max_iter = 1000 
abstol   = 1e-4 
reltol   = 1e-2 

# hyperparameter-update 
aug_L_hp(( sig_f, l, sig_n )) = aug_L(( sig_f, l, sig_n, dx, ξ, Θ, y, z, λ, ρ ))

σ0     = [1.0, 1.0, 0.1]
result = optimize(aug_L_hp, σ0)
println("aug_L_hp min = ", result.minimizer) 

# assign hyperparameters 
sig_f = result.minimizer[1] 
l     = result.minimizer[2] 
sig_n = result.minimizer[3] 

## ============================================ ##
# x-update 
aug_L_ξ(ξ) = aug_L(( sig_f, l, sig_n, dx, ξ, Θ, y, z, λ, ρ ))

σ0 = ξ
result = optimize(log_Z, σ0) 
println("minimizer = ", result.minimizer) 

# assign ξ
ξ = result.minimizer 

## ============================================ ##
# z-update

# shrinkage 
function shrinkage(x, kappa) 

    z = 0*x ; 
    for i = 1:length(x) 
        z[i] = max( 0, x[i] - kappa ) - max( 0, -x[i] - kappa ) 
    end 

    return z 
end 

z_old = z 
ξ_hat = α*ξ + ( 1 .- α*z_old )
z = shrinkage(ξ_hat + y, λ/ρ)

## ============================================ ##
# y-update 

y += ( ξ_hat .- z ) 
y += ρ*( ξ-z )



