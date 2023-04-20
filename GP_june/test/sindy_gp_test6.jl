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
# select ODE function 

function lorenz(du, (x,y,z), (σ,ρ,β), t)

    du[1] = dx = σ * ( y - x ) 
    du[2] = dy = x * ( ρ - z ) - y 
    du[3] = dz = x * y - β * z  

    return du 
end 

function predator_prey(dx, (x1,x2), (a,b,c,d), t)

    # control input 
    u = 2*sin(t) + 2*sin(t/10) 

    dx[1] = a*x1 - b*x1*x2 + u^2 * 0 
    dx[2] = -c*x2 + d*x1*x2  

    return dx 
end 

function ode_sine(dx, x, p, t)
    dx[1] = -1/4 * sin(x[1])  
    dx[2] = -1/2 * x[2] 
    return dx 
end 

## ============================================ ##
# get measurements 

# initial conditions and parameters 
fn     = predator_prey 
x0     = [ 1.0; 0.5 ]  
p      = [ 10.0, 28.0, 8/3, 2.0 ] 
n_vars = size(x0, 1) 
tf     = 100      
ts     = (0.0, tf)   
dt     = 0.1 
# u_fn(t) = 2*sin(t) + 2*sin(t/10) 

# solve ODE 
prob = ODEProblem(fn, x0, ts, p) 
sol  = solve(prob, saveat = dt) 
# sol  = solve(prob) 

# extract variables --> measurements 
sol_total = sol 
x = sol.u ; x = mapreduce(permutedims, vcat, x) 
t = sol.t 
u = 2*sin.(t) + 2*sin.(t/10) 

plt_static = plot( 
    sol, 
    # idxs   = (1,2,3), 
    title  = "Dynamics" 
    )


## ============================================ ##
# animated plot (interpolation)

# plot 
plt_anim  = plot(
    1,
    xlim = (0, tf), 
    # xlim   = (-30, 30),
    # ylim   = (-30, 30),
    # zlim   = (0, 50),
    title  = "Animation",
    legend = false,
    marker = 2, 
    )

# init animation and IC 
a  = Animation()	
c  = theme_palette(:auto) 
colors = [] 
for i = 1:n_vars 
    push!(colors, c[i])
end 
colors = colors[:,:]' 

# loop 
for i in 1:tf 

    #  time interval 
    ts = (i-1, i) 

    # solve ODE 
    prob = ODEProblem(fn, x0, ts, p) 
    sol  = solve(prob, saveat = dt) 
    # sol  = solve(prob) 

    # plot and save frame 
    # plot!(plt_anim, sol, idxs = (1,2,3), c = c, xlim = (-30, 30))
    plot!( plt_anim, sol, c = colors, xlim = (0, tf) )
    plt = plot( plt_static, plt_anim, layout = (2,1), size = [600 1000] )
    frame(a, plt)

    # next iter 
    x0 = sol.u[end]

end
	
plt_gif = gif(a, fps = 10)


## ============================================ ## 
# derivatives: finite differencing --> mapreduce x FIRST 

# extract variables --> measurements 
sol = sol_total 
x = sol.u ; x = mapreduce(permutedims, vcat, x) 
t = sol.t 

# (forward) finite difference 
dx_fd = 0*x 
for i = 1 : length(t)-1
    dx_fd[i,:] = ( x[i+1,:] - x[i,:] ) / ( t[i+1] - t[i] )
end 
dx_fd[end,:] = dx_fd[end-1,:] 

# true derivatives 
dx_true = 0*x
z = zeros(n_vars) 
for i = 1 : length(t) 
    dx_true[i,:] = fn( z, x[i,:], p, t[i] ) 
end 

# error 
dx_err  = dx_true - dx_fd 

## ============================================ ##
# plot derivatives 

plot_array = Any[] 
for j in 1 : n_vars
    plt = plot(t, dx_true[:,j], 
        title = "Axis $j", label = "true" ) 
        plot!(t, dx_fd[:,j], ls = :dash, label = "finite diff" )
  push!( plot_array, plt ) 
end

plot(plot_array ... , 
    layout = (n_vars, 1), 
    size = [600 n_vars*300], 
    plot_title = "Derivatives" )


## ============================================ ##
# SINDy alone 

λ = 0.1 

dx = dx_true 

# sindy 
Ξ_true = SINDy_c( x, u, dx_true, λ )
Ξ_fd   = SINDy_c( x, u, dx_fd, λ ) 

Ξ_true = SINDy( x, dx_true, λ )
Ξ_fd   = SINDy( x, dx_fd, λ ) 


## ============================================ ##
# validation 

# initial conditions and parameters 
fn     = predator_prey 
x0     = [ 1.0; 0.5 ]  
p      = [ 10.0, 28.0, 8/3, 2.0 ] 
n_vars = size(x0, 1) 
tf     = 20  
t0     = 0     
ts     = (t0, tf)   
dt     = 0.1 
# u_fn(t) = 2*sin(t) + 2*sin(t/10) 

# solve ODE 
prob = ODEProblem(fn, x0, ts, p) 
sol  = solve(prob, saveat = dt) 
# sol  = solve(prob) 
sol_train = sol 

# extract variables --> measurements 
x_train = sol.u ; x_train = mapreduce(permutedims, vcat, x_train) 
t_train = sol.t 
u_train = 2*sin.(t_train) + 2*sin.(t_train/10) 

# ----------------------- #
# generate validation data and predicts with SINDy solution 

# propagate truth 
x0 = x[end,:]
t0 = tf 
tf = 40 
ts = (t0, tf)

# solve ODE 
prob = ODEProblem(fn, x0, ts, p) 
sol  = solve(prob, saveat = dt) 
sol_validate = sol 

# sol  = solve(prob) 
x_validate = sol.u ; x_validate = mapreduce(permutedims, vcat, x_validate) 
t_validate = sol.t 
u_validate = 2*sin.(t_validate) + 2*sin.(t_validate/10) 

# ----------------------- #
# SINDy predict 

n_vars = size( x_train , 2)
poly_order = n_vars 

Θ_true_fn(x, p, t) = pool_data(x, n_vars, poly_order) * Ξ_true 
Θ_true_fn(x0', p, t)
ΘΞ_fn(x, p, t) = pool_data(x, n_vars, poly_order) * p 
ΘΞ_fn(x0', Ξ_true, t)

# solve ODE 
prob = ODEProblem(ΘΞ_fn, x0[:,:]', ts, Ξ_true) 
sol  = solve(prob, saveat = dt) 
sol_validate_sindy_true = sol 

# solve ODE 
prob = ODEProblem(ΘΞ_fn, x0[:,:]', ts, Ξ_fd) 
sol  = solve(prob, saveat = dt) 
sol_validate_sindy_fd = sol 


# validation plot 
plt_train_validate = plot( 
    sol_train, 
    # idxs   = (1,2,3), 
    label = "training", 
    )
    plot!(sol_validate, label = "validation")
    plot!(sol_validate_sindy_true, label = "sindy (true)", linestyle = :dash, xlim = (0, tf)    )
    plot!(sol_validate_sindy_fd, label = "sindy (fd)", linestyle = :dot, xlim = (0, tf), title = "Training and Validation" 
    )

## ============================================ ##
# smooth derivatives GP --> SINDy 

# ICs  
σ_0    = [1.0, 1.0, 0.1]  
σ_f, l, σ_n = σ_0 

# GP derivatives 
dx_GP    = 0 * dx_true 
std_post = 0 * dx_true 
for j = 1:n_vars 
    dx_GP[:,j], Σ = post_dist(( t, dx_fd[:,j], t, σ_f, l, σ_n ))
    cov_post = diag(Σ)
    std_post[:,j] = sqrt.(cov_post) 
end 
Ξ_sindy_GP   = SINDy( x, dx_GP, λ ) 
display(Ξ_sindy_GP) 

plot_array = Any[] 
for j in 1 : n_vars
    plt = plot(t, dx_GP[:,j] - dx_true[:,j], 
        title = "Axis $j error", label = "true" ) 
    push!( plot_array, plt ) 
end

plot(plot_array ... , 
    layout = (n_vars, 1), 
    size = [600 n_vars*300], 
    plot_title = "GP - Truth Derivatives" )


## ============================================ ##
# SINDy + GP + ADMM 

# # truth 
# hist_true = Hist( [], [], [], [], [] ) 
# @time z_true, hist_true = sindy_gp_admm( x, dx_true, λ, hist_true ) 
# display(z_true) 

# λ = 0.00001 * 0  

# finite difference 
hist_fd = Hist( [], [], [], [], [] ) 
@time z_fd, hist_fd = sindy_gp_admm( x, dx_fd, λ, hist_fd ) 
display(z_fd) 











## ============================================ ##
# sandbox 
## ============================================ ##
# SINDy-GP-LASSO, f_hp_opt 

x = sol_total.u ; x = mapreduce(permutedims, vcat, x) 
u = 2*sin.(t) + 2*sin.(t/10) 

n_vars = size( [x u], 2 )
poly_order = n_vars 

# construct data library 
Θx = pool_data( [x u], n_vars, poly_order) 

# first cut - SINDy 
Ξ_true = sparsify_dynamics( Θx, dx_true, λ, n_vars-1 ) 
 

## ============================================ ##
# each step 

z_soln = 0 * Ξ_true 

# ADMM stuff 
ρ = 1.0 
λ = 0.1 * 0 
α = 1.0 

# deal with state j 
# for j = 1 : n_vars 
j = 1 

    # initial loss function vars 
    ξ  = 0 * Ξ_true[:,j] 
    dx = dx_fd[:,j] 

    # assign for f_hp_opt 
    f_hp(ξ, σ_f, l, σ_n) = f_obj(( σ_f, l, σ_n, dx, ξ, Θx ))

    # l1 norm 
    g(z) = λ * sum(abs.(z)) 

    # ----------------------- #
    # admm!!! 

    hist_hp_opt = Hist( [], [], [], [], [] ) 

    n = length(ξ)
    # @time x_hp_opt, z_hp_opt, hist_hp_opt, k  = lasso_admm_hp_opt( f_hp, g, n, λ, ρ, α, hist_hp_opt ) 

## ============================================ ##
    # ----------------------- #
    # lasso_admm_hp_opt 
    
    # define constants 
    max_iter = 1000  
    abstol   = 1e-4 
    reltol   = 1e-2           # save matrix-vector multiply 

    # ADMM solver 
    x = zeros(n) ; z = zeros(n) ; u = zeros(n) 

    # initial hyperparameters 
    σ_f0 = 1.0 ; σ_f = σ_f0 ; 
    l_0  = 1.0 ; l   = l_0  ; 
    σ_n0 = 0.1 ; σ_n = σ_n0 ; 

    # bounds 
    lower = [0.0, 0.0, 0.0]  
    upper = [Inf, Inf, Inf] 

    # augmented Lagrangian (scaled form) 
    aug_L(x, σ_f, l, σ_n, z, u) = f_hp(x, σ_f, l, σ_n) + g(z) + ρ/2 .* norm( x - z + u )^2 
    
    # counter 
    iter = 0 

    # begin iterations 
    # for k = 1 : max_iter 
    k = 1 

        # increment counter 
        iter += 1 

        # ----------------------- #
        # x-update (optimization) 

        # optimization 
        f_opt(x) = aug_L(x, σ_f, l, σ_n, z, u) 
        od       = OnceDifferentiable( f_opt, x ; autodiff = :forward ) 
        result   = optimize( od, x, LBFGS() ) 
        x        = result.minimizer 

        # ----------------------- # 
        # hp-update (optimization) 

        σ_0    = [σ_f, l, σ_n]  
        hp_opt(( σ_f, l, σ_n )) = aug_L(x, σ_f, l, σ_n, z, u) 
        od     = OnceDifferentiable( hp_opt, σ_0 ; autodiff = :forward ) 
        result = optimize( od, lower, upper, σ_0, Fminbox(LBFGS()) ) 
        
        # assign optimized hyperparameters 
        σ_f = result.minimizer[1] 
        l   = result.minimizer[2] 
        σ_n = result.minimizer[3] 
        
        # ----------------------- #
        # z-update (soft thresholding) 

        z_old = z 
        x_hat = α*x + (1 .- α)*z_old 
        z     = shrinkage(x_hat + u, λ/ρ) 

        # ----------------------- #
        # u-update 

        u += (x_hat - z) 

        # ----------------------- #
        # diagnostics + termination checks 

        p = f(x, σ_f, l, σ_n) + g(z)   
        push!( hist.objval, p )
        push!( hist.r_norm, norm(x - z) )
        push!( hist.s_norm, norm( -ρ*(z - z_old) ) )
        push!( hist.eps_pri, sqrt(n)*abstol + reltol*max(norm(x), norm(-z)) ) 
        push!( hist.eps_dual, sqrt(n)*abstol + reltol*norm(ρ*u) ) 

        if hist.r_norm[k] < hist.eps_pri[k] && hist.s_norm[k] < hist.eps_dual[k] 
            break 
        end 

    # end 


    # ----------------------- #

    # solution residuals 
    println( "z_hp_opt - ξ_true = " ) 
    display( z_hp_opt - Ξ_true[:,j] ) 
    println( "SINDy - ξ_true = " )
    display( Ξ[:,j] - Ξ_true[:,j] )

    z_soln[:,j] = z_hp_opt 

# end 


## ============================================ ##
# ----------------------- #
# objective function 

z_soln = 0 * Ξ 

# ADMM stuff 
ρ = 1.0 
λ = 0.1 * 0 
α = 1.0 

# deal with state j 
# for j = 1 : n_vars 
j = 1 

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

# end 


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



    