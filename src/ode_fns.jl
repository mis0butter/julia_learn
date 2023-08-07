using DifferentialEquations


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

export predator_prey_forcing
function predator_prey_forcing(dx, (x1,x2), (a,b,c,d), t; u = 2sin(t) + 2sin(t/10))

    dx[1] = a*x1 - b*x1*x2 + u^2 
    dx[2] = -c*x2 + d*x1*x2  

    return dx 
end 

export ode_sine 
function ode_sine(dx, x, p, t)
    dx[1] = 1/2*sin(x[1])  
    # dx[2] = -1/2 * x[2] 
    return dx 
end 



# Constants, I do like that I do not have to parse them manually to ode78
g           = 9.81   # Acceleration due to gravity in m/s^2
l           = 1.0     # Length of pendulum 1 in metres

export pendulum
function pendulum(dx, x, p, t)

    θ  = x[1] 
    dθ = x[2] 

	# The double pendulum equations
    # dx = [ 0.0; 0.0]
    dx[1] = dθ 
    dx[2] = -( g / l ) * cos(θ)

    # Return the derivatives as a vector
	return dx
end


## ============================================ ##
# solve ODE problem 

export solve_ode 
function solve_ode(fn, x0, str, p, ts, dt, plot_option)

    # x0, str, p, ts, dt = init_params(fn) 

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

    return t, x 

end 


## ============================================ ##
# solve ODE problem, compute derivatives and plot states 

export ode_states 
function ode_states(fn, plot_option, fd_method)

    x0, str, p, ts, dt = init_params(fn) 
    t, x = solve_ode(fn, x0, str, p, ts, dt, plot_option) 

    # ----------------------- #
    # derivatives 
    dx_fd   = fdiff(t, x, fd_method)    # finite difference 
    dx_true = dx_true_fn(t, x, p, fn)   # true derivatives 
    # dx_tv   = dx_tv_fn(x)               # variational derivatives 
    # dx_gp   = dx_gp_fn(t, dx_fd)        # gaussian process derivatives 

    # error 
    dx_err  = dx_true - dx_fd 

    # plot derivatives 
    if plot_option == 1 
        plot_deriv(t, dx_true, dx_fd, dx_tv, str) 
    end 

    return x0, dt, t, x, dx_true, dx_fd, p 

end 


## ============================================ ##
# generate predicts / validation data 

export validate_data 
function validate_data(t_test, x_test, dx_fn, dt)


    n_vars = size(x_test,2) 
    x0     = [ x_test[1] ] 
    if n_vars > 1 
        x0 = x_test[1,:] 
    end 

    # dt    = t_test[2] - t_test[1] 
    tspan = (t_test[1], t_test[end])
    prob  = ODEProblem(dx_fn, x0, tspan) 

    # solve the ODE
    sol = solve(prob, saveat = dt)
    # sol = solve(prob,  reltol = 1e-8, abstol = 1e-8)
    x_validate = sol.u ; 
    x_validate = mapreduce(permutedims, vcat, x_validate) 
    t_validate = sol.t 

    return t_validate, x_validate 

end 


## ============================================ ##
# derivatives: finite difference  

export fdiff 
function fdiff(t, x, fd_method) 

    # forward finite difference 
    if fd_method == 1 

        dx_fd = 0*x 
        for i = 1 : length(t)-1
            dx_fd[i,:] = ( x[i+1,:] - x[i,:] ) / ( t[i+1] - t[i] )
        end 

        # deal with last index 
        dx_fd[end,:] = ( x[end,:] - x[end-1,:] ) / ( t[end] - t[end-1] )

    # central finite difference 
    elseif fd_method == 2 

        dx_fd = 0*x 
        for i = 2 : length(t)-1
            dx_fd[i,:] = ( x[i+1,:] - x[i-1,:] ) / ( t[i+1] - t[i-1] )
        end 

        # deal with 1st index 
        i = 1 
        dx_fd[i,:] = ( x[i+1,:] - x[i,:] ) / ( t[i+1] - t[i] )

        # deal with last index 
        dx_fd[end,:] = ( x[end,:] - x[end-1,:] ) / ( t[end] - t[end-1] )

    # backward finite difference 
    else 

        dx_fd = 0*x 
        for i = 2 : length(t)
            dx_fd[i,:] = ( x[i,:] - x[i-1,:] ) / ( t[i] - t[i-1] )
        end 

        # deal with 1st index 
        i = 1 
        dx_fd[i,:] = ( x[i+1,:] - x[i,:] ) / ( t[i+1] - t[i] )

    end 

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


## ============================================ ##
# 

export build_dx_fn 
function build_dx_fn(poly_order, z_fd)

    # get # states 
    n_vars = size( z_fd, 2 ) 

    # define pool_data functions 
    fn_vector = pool_data_vecfn(n_vars, poly_order) 

    # numerically evaluate each function at x and return a vector of numbers
    𝚽( x, fn_vector ) = [ f(x) for f in fn_vector ]

    # create vector of functions, each element --> each state 
    dx_fn_vec = Vector{Function}(undef,0) 
    for i = 1:n_vars 
        # define the differential equation 
        push!( dx_fn_vec, (x,p,t) -> dot( 𝚽( x, fn_vector ), z_fd[:,i] ) ) 
    end 

    dx_fn(x,p,t) = [ f(x,p,t) for f in dx_fn_vec ] 

    return dx_fn 

end 



