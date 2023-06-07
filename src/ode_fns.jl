
## ============================================ ##
# ODE functions 

export lorenz 
function lorenz(du, (x,y,z), (σ,ρ,β), t)

    du[1] = dx = σ * ( y - x ) 
    du[2] = dy = x * ( ρ - z ) - y 
    du[3] = dz = x * y - β * z  

    return du 
end 

export predator_prey 
function predator_prey(dx, (x1,x2), (a,b,c,d), t; u = 2sin(t) + 2sin(t/10))

    dx[1] = a*x1 - b*x1*x2 + u^2 * 0 
    dx[2] = -c*x2 + d*x1*x2  

    return dx 
end 

export ode_sine 
function ode_sine(dx, x, p, t)
    dx[1] = -1/4 * sin(x[1])  
    # dx[2] = -1/2 * x[2] 
    return dx 
end 


# ## ============================================ ##
# solve ODE problem, compute derivatives and plot states 

using DifferentialEquations

export ode_states 
function ode_states(fn, plot_option)

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
    dx_tv   = dx_tv_fn(x)               # variational derivatives 
    # dx_gp   = dx_gp_fn(t, dx_fd)        # gaussian process derivatives 

    # error 
    dx_err  = dx_true - dx_fd 

    # plot derivatives 
    if plot_option == 1 
        plot_deriv(t, dx_true, dx_fd, dx_tv, str) 
    end 

    return t, x, dx_true, dx_fd, dx_tv 

end 


## ============================================ ##
# derivatives: finite difference  

export fdiff 
function fdiff(t, x) 

    # (forward) finite difference 
    dx_fd = 0*x 
    for i = 1 : length(t)-1
        dx_fd[i,:] = ( x[i+1,:] - x[i,:] ) / ( t[i+1] - t[i] )
    end 
    dx_fd[end,:] = dx_fd[end-1,:] 

    return dx_fd 

end 


## ============================================ ##
# derivatives: truth 

export dx_true_fn 
function dx_true_fn(t, x, p, fn)

    # true derivatives 
    dx_true = 0*x
    n_vars  = size(x, 2) 
    z       = zeros(n_vars) 

    for i = 1 : length(t) 
        dx_true[i,:] = fn( z, x[i,:], p, t[i] ) 
    end 

    return dx_true 

end 


## ============================================ ##
# derivatives: variational 

using NoiseRobustDifferentiation

export dx_tv_fn 
function dx_tv_fn(x) 

    dx_tv  = 0*x 
    n_vars = size(x, 2)

    for i = 1:n_vars 
        dx = x[2,i] - x[1,i] 
        dx_tv[:,i] = tvdiff(x[:,i], 100, 0.2, dx=dx)
    end 

    return dx_tv 

end 


## ============================================ ##
# derivatives: gaussian process (smoothing) 

using Optim 

export dx_gp_fn 
function dx_gp_fn(t, dx) 

    x_train = t 
    y_train = dx 
    x_test  = t 

    f_hp( σ_f, l, σ_n ) = log_p( σ_f, l, σ_n, x_train, y_train, x_train*0 )

    # ----------------------- # 
    # hp-update (optimization) 
    
    # bounds 
    lower = [0.0, 0.0, 0.0]  
    upper = [Inf, Inf, Inf] 
    σ_0   = [1.0, 1.0, 0.1]  

    od     = OnceDifferentiable( f_hp, σ_0 ; autodiff = :forward ) 
    result = optimize( od, lower, upper, σ_0, Fminbox( LBFGS() ) ) 
        
    # assign optimized hyperparameters 
    σ_f = result.minimizer[1] 
    l   = result.minimizer[2] 
    σ_n = result.minimizer[3] 

    μ_post, Σ_post = post_dist( x_train, y_train, x_test, σ_f, l, σ_n )

    return μ_post 

end 






