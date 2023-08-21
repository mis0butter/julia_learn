using DifferentialEquations


## ============================================ ##


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


function validate_data(t_test, x_test, dx_fn, dt)


    n_vars = size(x_test, 2) 
    x0     = [ x_test[1] ] 
    if n_vars > 1 
        x0 = x_test[1,:] 
    end 

    # dt    = t_test[2] - t_test[1] 
    tspan = (t_test[1], t_test[end])
    prob  = ODEProblem(dx_fn, x0, tspan) 

    # solve the ODE
    sol   = solve(prob, saveat = dt)
    # sol = solve(prob,  reltol = 1e-8, abstol = 1e-8)
    x_validate = sol.u ; 
    x_validate = mapreduce(permutedims, vcat, x_validate) 
    t_validate = sol.t 

    return t_validate, x_validate 

end 


## ============================================ ##


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


function build_dx_fn(poly_order, z_fd)

    # get # states 
    n_vars = size( z_fd, 2 ) 

    # define pool_data functions 
    fn_vector = pool_data_vecfn_test(n_vars, poly_order) 

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
