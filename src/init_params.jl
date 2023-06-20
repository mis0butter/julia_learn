

export init_params 
function init_params(fn) 

    # initial conditions and parameters 
    if fn == lorenz 
        x0  = [ 1.0; 0.5; 0 ]
        str = "lorenz" 
    elseif fn == ode_sine 
        x0  = [ 1.0 ]  
        str = "ode_sine" 
    elseif fn == predator_prey 
        # x0  = [ 10; 10 ] 
        x0  = 10*round.( rand(2), digits=2 )
        str = "predator_prey" 
    end 
    # p      = [ 10.0, 28.0, 8/3, 2.0 ] 
    # p      = [ 1.1, 0.4, 0.1, 0.4 ] 
    p      = [ 1.5, 1, 3, 1 ] 
    n_vars = size(x0, 1) 
    tf     = 14 
    ts     = (0.0, tf) 
    dt     = 0.1 

    return x0, str, p, ts, dt

end 