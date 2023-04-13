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

function lorenz(du, (x,y,z), (σ,ρ,β), t)

    du[1] = dx = σ * ( y - x ) 
    du[2] = dy = x * ( ρ - z ) - y 
    du[3] = dz = x * y - β * z  

    return du 
end 

# initial conditions and timespan 
x0 = [1; 0; 0]  ;   n_vars = size(x0, 1) 
tf = 100        ;   ts = (0.0, tf)   
p  = [ 10.0, 28.0, 8/3 ] 

# solve ODE 
prob = ODEProblem(lorenz, x0, ts, p) 
sol  = solve(prob, saveat = 0.01) 

# extract variables --> measurements 
x = sol.u ; x = mapreduce(permutedims, vcat, x) 
t = sol.t 

plt_static = plot( 
    sol, 
    xlim   = (-30, 30),
    ylim   = (-30, 30),
    zlim   = (0, 50),
    idxs   = (1,2,3), 
    legend = false, 
    title  = "Lorenz Attractor" 
    )


## ============================================ ##
# animated plot (interpolation)

# plot 
plt_anim  = plot3d(
    1,
    xlim   = (-30, 30),
    ylim   = (-30, 30),
    zlim   = (0, 50),
    title  = "Animation",
    legend = false,
    marker = 2, 
    )

# init animation and IC 
a  = Animation()	
x0 = [1.0, 0, 0]
c  = theme_palette(:auto) 

# loop 
for i in 1:tf 

    #  time interval 
    ts = (i-1, i) 

    # solve ODE 
    prob = ODEProblem(lorenz, x0, ts, p) 
    sol  = solve(prob, saveat = .01) 

    # plot and save frame 
    plot!(plt_anim, sol, idxs = (1,2,3), c = c, xlim = (-30, 30))
    plt = plot( plt_static, plt_anim, layout = (2,1), size = [600 1000] )
    frame(a, plt)

    # next iter 
    x0 = sol.u[end]

end
	
plt_gif = gif(a, fps = 5)


## ============================================ ## 
# derivatives: finite differencing --> mapreduce x FIRST 

# extract variables --> measurements 
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
z = zeros(3) 
for i = 1 : length(t) 
    dx_true[i,:] = lorenz( z, x[i,:], p, 0 ) 
end 

# error 
dx_err  = dx_true - dx_fd 

# plot 
plt_1 = plot(t, dx_true[:,1], 
    title = "Axis 1", label = "true" ) 
    plot!(t, dx_fd[:,1], ls = :dash, label = "finite diff" )
plt_2 = plot(t, dx_true[:,2], 
    title = "Axis 2") 
    plot!(t, dx_fd[:,2], ls = :dash, legend = false)
plt_3 = plot(t, dx_true[:,3], 
    title = "Axis 3", xlabel = "Time (s)") 
    plot!(t, dx_fd[:,3], ls = :dash, legend = false)

plt_dx = plot(plt_1, plt_2, plt_3, layout = (3,1), size = [600 1000], plot_title = "Derivatives")


## ============================================ ##

λ = 0.1 

# sindy 
Ξ_sindy_true = SINDy( x, dx_true, λ ) 
Ξ_sindy_fd   = SINDy( x, dx_fd, λ ) 
## ============================================ ##

# truth 
hist_true = Hist( [], [], [], [], [] ) 
@time z_true, hist_true = sindy_gp_admm( x, dx_true, λ, hist_true ) 

# finite difference 
hist_fd = Hist( [], [], [], [], [] ) 
@time z_fd, hist_fd = sindy_gp_admm( x, dx_fd, λ, hist_fd ) 







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



    