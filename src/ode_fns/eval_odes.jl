using DifferentialEquations

## ============================================ ##

export ode_train_test 
function ode_train_test( fn ) 

    x0, dt, t, x_true, dx_true, dx_fd, p = ode_states(fn, 0, 2) 

    # noise 
    noise = 0.01 
    x_noise  = x_true + noise*randn( size(x_true, 1), size(x_true, 2) )
    dx_noise = dx_true + noise*randn( size(dx_true, 1), size(dx_true, 2) )

    u = [] 
    for i = 1 : length(t) 
        # push!( u, [ 1/2*sin(t[i]), cos(t[i]) ] ) 
        push!( u, 2sin(t[i]) + 2sin(t[i]/10) ) 
    end 
    # u = vv2m(u) 

    # split into training and test data 
    test_fraction = 0.2 
    portion       = 5 
    u_train,        u_test        = split_train_test(u,        test_fraction, portion) 
    t_train,        t_test        = split_train_test(t,        test_fraction, portion) 
    x_train_true,   x_test_true   = split_train_test(x_true,   test_fraction, portion) 
    dx_train_true,  dx_test_true  = split_train_test(dx_true,  test_fraction, portion) 
    x_train_noise,  x_test_noise  = split_train_test(x_noise,  test_fraction, portion) 
    dx_train_noise, dx_test_noise = split_train_test(dx_noise, test_fraction, portion) 

    data_train = data_struct( t_train, u_train, x_train_true, dx_train_true, x_train_noise, dx_train_noise ) 
    data_test  = data_struct( t_test, u_test, x_test_true, dx_test_true, x_test_noise, dx_test_noise) 

    return data_train, data_test 
end 


## ============================================ ##

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

export validate_data 
function validate_data(t_test, xu_test, dx_fn, dt)


    n_vars = size(xu_test, 2) 
    x0     = [ xu_test[1] ] 
    if n_vars > 1 
        x0 = xu_test[1,:] 
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
# (5) Function to integrate an ODE using forward Euler integration.

export integrate_euler 
function integrate_euler(dx_fn, x₀, p, T; u = false)
    # TODO: Euler integration consists of setting x(t + δt) ≈ x(t) + δt * ẋ(t, x(t), u(t)).
    #       Returns x(T) given x(0) = x₀.

    xt  = x₀ 

    if u == false 

        t  = 0 : 0.01 : T 
        n  = length(t) 

        for i = 1 : n 
            xt += δt * dx_fn( xt, t[i] ) 
        end     

    else 

        n  = length(u) 
        dt = T / n 
        t  = 0 : dt : T     

        for i = 1 : n 
            xt += δt * dx_fn( xt, t[i], u[i] ) 
        end 
    
    end 

    return xt 
end

## ============================================ ##

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

export build_dx_fn 
function build_dx_fn(poly_order, x_vars, u_vars, z_fd) 

    n_vars = x_vars + u_vars 

    # define pool_data functions 
    fn_vector = pool_data_vecfn_test(n_vars, poly_order) 

    # numerically evaluate each function at x and return a vector of numbers
    𝚽( xu, fn_vector ) = [ f(xu) for f in fn_vector ]

    # create vector of functions, each element --> each state 
    dx_fn_vec = Vector{Function}(undef,0) 
    for i = 1 : x_vars 
        # define the differential equation 
        push!( dx_fn_vec, (xu,p,t) -> dot( 𝚽( xu, fn_vector ), z_fd[:,i] ) ) 
    end 

    dx_fn(xu,p,t) = [ f(xu,p,t) for f in dx_fn_vec ] 

    return dx_fn 

end 


