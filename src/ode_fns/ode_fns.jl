using DifferentialEquations


## ============================================ ##
# ODEs 

include("odes.jl")
export lorenz, ode_sine 
export predator_prey, predator_prey_forcing  
export pendulum, double_pendulum 
export unicycle, dyn_car 
export quadcopter, rotation_euler 


## ============================================ ##
# solve ODE problem 

include("eval_odes.jl")
export solve_ode, ode_states, validate_data
export dx_true_fn, build_dx_fn 


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


