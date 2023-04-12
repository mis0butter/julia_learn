struct Hist 
    objval 
    r_norm 
    s_norm 
    eps_pri 
    eps_dual 
end 

using DifferentialEquations 
using GP_june 
using LinearAlgebra 
using ForwardDiff 
using Optim 
using Plots 


## ============================================ ##
# ODE function 

function ODE_test(dx, x, p, t)
    dx[1] = -1/4 * sin(x[1]) 
    dx[2] = -1/2 * x[2] 
    return dx 
end 

# initial conditions and timespan 
x0 = [1; 1] ;       n_vars = size(x0, 1) 
ts = (0.0, 10.0) ;  dt = 0.1 

# solve ODE 
prob = ODEProblem(ODE_test, x0, ts) 
sol  = solve(prob, saveat = dt) 

using Plots 
p_ode = plot(sol, label = [ "x1 (true)" "x2 (true)"], title = "x" ) 


## ============================================ ## 
# derivatives: finite differencing --> mapreduce x FIRST 

# extract variables --> measurements 
x = sol.u ; x = mapreduce(permutedims, vcat, x) 
t = sol.t 

# (forward) finite difference 
dx_fd = 0*x 
for i = 1 : length(t)-1
    dx_fd[i,:] = ( x[i+1,:] - x[i,:] ) / dt 
end 
dx_fd[end,:] = dx_fd[end-1,:] 

# true derivatives 
dx_true = 0*x
for i = 1 : length(t) 
    dx_true[i,:] = ODE_test( [0.0, 0.0], x[i,:], 0.0, 0.0 ) 
end 

# error 
dx_err  = dx_true - dx_fd 


## ============================================ ##
# plot truth and finite diff dx 

p_dx = plot(t, dx_true, 
    lw = 2, xlabel = "t", title = "dx", label = [ "dx1 (true)" "dx2 (true)" ]) 
plot!(p_dx, t, dx_fd, 
    ls = :dot, lw = 2, label = [ "dx1 (diff)" "dx2 (diff)" ]) 

# plot dx err 
p_dx_err = plot(t, dx_err, 
    lw = 2, title = "dx (err) = true - diff", label = [ "dx1 (err)" "dx2 (err)" ])

# plot all 
p_ode_dx = plot(p_ode, p_dx, p_dx_err, layout = (3,1), 
    size = [ 600, 800 ], plot_title = " x and dx " )
  

## ============================================ ##

λ = 0.1 

hist_hp_opt = Hist( [], [], [], [], [] ) 

@time z_soln, hist_hp_opt = sindy_gp_admm( x, dx_fd, λ, hist_hp_opt ) 








## ============================================ ##
# sandbox 
## ============================================ ##
# SINDy-GP-LASSO, f_hp_opt 

#define inputs 
n_vars     = 2  
poly_order = n_vars 

# construct data library 
Θx   = pool_data(x, n_vars, poly_order) 

# sparsification knob 
λ = 0.1 

# first cut - SINDy 
Ξ_true = sparsify_dynamics(Θx, dx_true, λ, n_vars) 
Ξ      = sparsify_dynamics(Θx, dx_fd, λ, n_vars) 

# ----------------------- #
# objective function 

z_soln = 0 * Ξ 

# ADMM stuff 
ρ = 1.0 
λ = 0.1 
α = 1.0 

# deal with state j 
for j = 1 : n_vars 

    # initial loss function vars 
    ξ  = 0 * Ξ[:,j] 
    dx = dx_fd[:,j] 

    # assign for f_hp_opt 
    f_hp(ξ, σ_f, l, σ_n) = f_obj(( σ_f, l, σ_n, dx, ξ, Θx ))

    # l1 norm 
    g(z) = λ * sum(abs.(z)) 

    # ----------------------- #
    # admm!!! 

    hist_hp_opt = Hist( [], [], [], [], [] ) 

    n = length(ξ)
    @time x_hp_opt, z_hp_opt, hist_hp_opt, k  = lasso_admm_hp_opt( f_hp, g, n, λ, ρ, α, hist_hp_opt ) 

    # solution residuals 
    println( "z_hp_opt - ξ_true = " ) 
    display( z_hp_opt - Ξ_true[:,j] ) 
    println( "SINDy - ξ_true = " )
    display( Ξ[:,j] - Ξ_true[:,j] )

    z_soln[:,j] = z_hp_opt 

end 


## ============================================ ##
# SINDy-GP-LASSO, f_opt 

#define inputs 
n_vars     = 2  
poly_order = n_vars 

# construct data library 
Θx   = pool_data(x, n_vars, poly_order) 

# sparsification knob 
λ = 0.1 

# first cut - SINDy 
Ξ_true = sparsify_dynamics(Θx, dx_true, λ, n_vars) 
Ξ      = sparsify_dynamics(Θx, dx_fd, λ, n_vars) 

# ----------------------- #
# objective function 

z_soln = 0 * Ξ 

# ADMM stuff 
ρ = 1.0 
λ = 0.1 
α = 1.0 

# deal with state j 
for j = 1 : n_vars 

    # initial loss function vars 
    ξ  = 0 * Ξ[:,j] 
    dx = dx_fd[:,j] 

    # assign for f_hp_opt 
    f_hp(ξ, σ_f, l, σ_n) = f_obj(( σ_f, l, σ_n, dx, ξ, Θx ))

    # initial hyperparameters 
    σ_f = 1.0 
    l   = 1.0 
    σ_n = 0.1 

    # assign for f_opt 
    f(ξ) = f_obj(( σ_f, l, σ_n, dx, ξ, Θx ))

    # l1 norm 
    g(z) = λ * sum(abs.(z)) 

    # ----------------------- #
    # admm!!! 

    hist_hp_opt = Hist( [], [], [], [], [] ) 
    hist_opt    = Hist( [], [], [], [], [] ) 
    # hist_test   = Hist( [], [], [], [], [] ) 

    n = length(ξ)
    @time x_hp_opt, z_hp_opt, hist_hp_opt, k  = lasso_admm_hp_opt( f_hp, g, n, λ, ρ, α, hist_hp_opt ) 
    @time x_opt,    z_opt,    hist_opt,    k  = lasso_admm_opt( f, g, n, λ, ρ, α, hist_opt ) 
    # @time x_test,   z_test,   hist_test       = lasso_admm_test( f, g, n, λ, ρ, α, hist_test ) 

    # solution residuals 
    println( "z_opt - ξ_true = " )
    display( z_opt - Ξ_true[:,j] ) 
    println( "z_hp_opt - ξ_true = " )
    display( z_hp_opt - Ξ_true[:,j] ) 
    println( "SINDy - ξ_true = " )
    display( Ξ[:,j] - Ξ_true[:,j] )

    z_soln[:,j] = z_hp_opt 

end 


## ============================================ ##
# plot 

p_opt  = plot_admm(hist_opt) 
    plot!(plot_title = "ADMM Lasso (x-opt) \n state = $(j)" )
p_hp_opt = plot_admm(hist_hp_opt) 
    plot!(plot_title = "ADMM Lasso (x-hp-opt) \n state = $(j)")

fig = plot(p_opt, p_hp_opt, layout = (1,2), size = [800 800])



    