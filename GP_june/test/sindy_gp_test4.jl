struct Hist 
    objval 
    r_norm 
    s_norm 
    eps_pri 
    eps_dual 
end 

using Revise 
using DifferentialEquations 
using GP_june 
using LinearAlgebra 
using Zygote 
using Optim 


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
ts = (0.0, 10.0) ; 
dt = 0.1 ; 

# solve ODE 
prob = ODEProblem(ODE_test, x0, ts) ; 
sol  = solve(prob, saveat = dt) ; 

using Plots 
p1 = plot(sol) ; 


## ============================================ ## 
# finite differencing 

# extract variables 
x = sol.u ; 
t = sol.t ; 

# finite difference 
dx = 0*x ; 
for i = 1 : length(x)-1
    dx[i] = ( x[i+1] - x[i] ) / dt ; 
end 
dx[end] = dx[end-1] ; 

# true derivatives 
dx_true = 0*dx ; 
for i = 1 : length(x) 
    dx_true[i] = ODE_test([0.0, 0.0], x[i], 0.0, 0.0 ) ; 
end 

# convert vector of vectors into matrix 
dx_true = mapreduce(permutedims, vcat, dx_true) 
dx      = mapreduce(permutedims, vcat, dx) 

# add to plot 
p2 = plot(t, dx_true, lw = 2) ; 
    plot!(p2, t, dx, ls = :dot, lw = 2) 
    title!(p2, "True and Finite Difference Derivatives")
    xlabel!(p2, "t")
  
# plot all 
p3 = plot(p1, p2, layout = (2,1), size = [ 600, 800 ])
  
## ============================================ ##
# SINDy 

#define inputs 
n_vars     = 2 ; 
poly_order = n_vars ;

# construct data library 
Θx = pool_data(x, n_vars, poly_order) ; 

# sparsification knob 
λ = 0.1 ; 

# first cut - SINDy 
Ξ_true = sparsify_dynamics(Θx, dx_true, λ, n_vars)
Ξ      = sparsify_dynamics(Θx, dx, λ, n_vars)


## ============================================ ##
# setting up GP stuff 

# initial hyperparameters 
σ_f0 = 1.0 ; σ_f = σ_f0 ; 
l_0  = 1.0 ; l   = l_0  ; 
σ_n0 = 0.1 ; σ_n = σ_n0 ; 

# try first state 
ξ  = Ξ[:,1] 
dx = dx[:,1]  

log_p_hp(( σ_f, l, σ_n )) = log_p(( σ_f, l, σ_n, dx, dx, Θx*ξ ))

σ_0 = [σ_f0, l_0, σ_n0]
result = optimize(log_p_hp, σ_0) 
println("minimizer = ", result.minimizer) 


## ============================================ ##
# full augmented Lagrangian 

ρ = 1.0 
λ = 0.1 
y = 0*ξ 
z = 0*ξ
α = 1.0 

function aug_L(( σ_f, l, σ_n, dx, ξ, Θx, y, z, λ, ρ ))
    
    # kernel function 
    # k_fn(σ_f, l, xp, xq) = σ_f^2 * exp.( -1/( 2*l^2 ) * sq_dist(xp, xq) ) 

    # training kernel function 
    Ky = k_fn((σ_f, l, dx, dx)) 
    Ky += σ_n^2 * I 

    term  = 1/2*( dx - Θx*ξ )'*inv( Ky )*( dx - Θx*ξ ) 
    term += 1/2*log(det( Ky )) 
    term += λ*sum(abs.(z)) 
    term += y'*(ξ-z) 
    term += ρ/2*( norm(ξ-z) )^2 

    return 

end 

# test 
out = aug_L(( σ_f0, l_0, σ_n0, dx, ξ, Θx, y, z, λ, ρ ))
println("testing aug_L = ", out)

# test optimization, reassign function 
aug_L_hp(( σ_f, l, σ_n )) = aug_L(( σ_f, l, σ_n, dx, ξ, Θx, y, z, λ, ρ ))
aug_L_hp(( σ_f0, l_0, σ_n0 ))

# creating bounds 
lower = [0.0, 0.0, 0.0] 
upper = [Inf, Inf, Inf]

# σ_0 = [σ_f0, l_0, σ_n0]
# result = optimize( aug_L_hp, lower, upper, σ_0,  Fminbox(LBFGS()) ) 
# println("log_p min (LBFGS) = \n ", result.minimizer) 

# # assign hyperparameters 
# σ_f = result.minimizer[1] 
# l   = result.minimizer[2] 
# σ_n = result.minimizer[3] 

# result = optimize(aug_L_hp, σ_0)
# println("aug_L_hp min = ", result.minimizer) 


## ============================================ ##
# ADMM 

hist = Hist([], [], [], [], [])

ρ = 1.0 
λ = 0.1 
y = 0*ξ 
z = 0*ξ
α = 1.0 

max_iter = 10
abstol   = 1e-2
reltol   = 1e-2 

# σ_f = 1.0 
# l   = 1.0 
# σ_n = 0.1 

σ_0   = [σ_f0, l_0, σ_n0] 
# σ_0    = [ σ_f, l_0, σ_n ] * 1.1 
lower = [0.0, 0.0, 0.0] 
upper = [Inf, Inf, Inf]

iter = 0 

for k = 1:max_iter 

    iter += 1 

    # hyperparameter-update 
    aug_L_hp(( σ_f, l, σ_n )) = aug_L(( σ_f, l, σ_n, dx, ξ, Θx, y, z, λ, ρ ))

    σ_0     = [σ_f, l, σ_n] 
    result = optimize( aug_L_hp, lower, upper, σ_0,  Fminbox(LBFGS()) ) 
    println("log_p min (LBFGS) = \n ", result.minimizer) 

    # assign hyperparameters 
    σ_f = result.minimizer[1] 
    l   = result.minimizer[2] 
    σ_n = result.minimizer[3] 

    # ----------------------- #
    # x-update 
    
    aug_L_ξ(ξ) = aug_L(( σ_f, l, σ_n, dx, ξ, Θx, y, z, λ, ρ )) 
    
    println( "ξ norm = ", norm(ξ) )
    println( "aug_L_ξ = ", aug_L_ξ(ξ), "\n\n" )    

    σ_0 = ξ
    result = optimize(aug_L_ξ, σ_0) 

    # assign ξ
    ξ = result.minimizer 

    # ----------------------- #    
    # z-update

    aug_L_z(z) = aug_L(( σ_f, l, σ_n, dx, ξ, Θx, y, z, λ, ρ ))
    println( "z norm = ", norm(z) )
    println( "aug_L_z = ", aug_L_z(z), "\n\n" )    

    # σ_0 = z 
    # result = optimize(aug_L_ξ, z) 
    # println("minimizer = ", result.minimizer) 

    z_old = z 
    # ξ_hat = α*ξ + ( 1 .- α*z_old )
    v = ξ - 1/ρ * y 
    z = shrinkage(v, λ/ρ)

    # dumb pause 
    # readline() 

    # ----------------------- #
    # y-update 

    # y += ( ξ_hat .- z ) 
    y += ρ*( ξ-z )

    # ----------------------- #
    # diagnostics 

    n = length(ξ) ; u = y/ρ 
    push!( hist.r_norm, norm(ξ - z) )
    push!( hist.s_norm, norm( -ρ*(z - z_old) ) )
    push!( hist.eps_pri, sqrt(n)*abstol + reltol*max(norm(x), norm(-z)) ) 
    push!( hist.eps_dual, sqrt(n)*abstol + reltol*norm(ρ*u) ) 

    if hist.r_norm[k] < hist.eps_pri[k] && hist.s_norm[k] < hist.eps_dual[k] 
        println("converged!") 
        break 
    elseif hist.r_norm[k] > 100 
        println("norm(ξ - z) too big")
        break 
    end 

    if k == max_iter 
        println("reached max_iter = ", max_iter)
    end 

end 

p4 = plot(1:iter, hist.r_norm, title = "|r| = norm(ξ-z)")
p5 = plot(1:iter, hist.s_norm, title = "|s| = norm(-ρ(z-z_old))") 

p6 = plot(p4, p5, layout = (2,1), size = [600, 800])

