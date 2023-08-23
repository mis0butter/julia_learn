using Plots 

export init_params 
function init_params(fn) 

    # initial conditions and parameters 
    if fn == lorenz 
        x0  = [ 1.0; 0.5; 0 ]
        p   = [ 1.1, 0.4, 1, 0.4 ] 
        str = "lorenz" 

    elseif fn == ode_sine 
        x0  = [ 1.0 ]  
        p   = [ 1.1, 0.4, 1, 0.4 ] 
        str = "ode_sine" 

    elseif fn == predator_prey 
        # x0  = [ 1.0; 0.5 ]
        x0  = [ 10.0; 5.0 ] 
        p   = [ 1.1, 0.4, 1, 0.4 ] 
        # x0  = 10*round.( rand(2), digits=2 )
        # x0  = 0.5 .+ 0.25*rand(2) 
        # println("x0 = ", x0)
        str = "predator_prey" 

    elseif fn == pendulum 
        θ   = 0.0 ; dθ = 0.0  
        x0  = [ θ, dθ ]
        l = 1 ; m = 1  
        p   = [ l, m ] 
        str = "pendulum"

    elseif fn == double_pendulum

        θ₁  = 1.6 ; dθ₁ = 0.0 ; θ₂  = 2.2 ; dθ₂ = 0.0    
        x0 = [ θ₁, dθ₁, θ₂, dθ₂ ] 

        m1 = 1 ; m2 = 1 ; l1 = 1 ; l2 = 1 
        p  = [ l1, l2, m1, m2 ] 

        str = "double_pendulum"
        
    elseif fn == unicycle 

        x0  = [ 0, 0, 0.5, 0.5 ] 
        p   = zeros(4) 
        str = "double_pendulum" 

    elseif fn == dyn_car 

        x0 = [ 90.0000, 4.2500, 13.0000, 3.1416, 5.0000, 2.0000, 0, 0, 0, 0, 0, 0 ] 
        p = zeros(4) 
        str = "dyn_car" 

    elseif fn == quadcopter 
        
        x0  = ones(12) 
        p   = ones(4) 
        str = "quadcopter" 

    end 
    # p      = [ 10.0, 28.0, 8/3, 2.0 ] 
    # p      = [ 1.5, 1, 3, 1 ] 
    n_vars = size(x0, 1) 
    tf     = 10.0  
    ts     = (0.0, tf) 
    dt     = 0.1 

    # initial plotting stuff 
    plot_font = "Computer Modern" 
    fsize = 14 
    default(
        fontfamily = plot_font,
        linewidth = 2, 
        # framestyle = :box, 
        label = nothing, 
        grid  = false, 
        legend = false, 
        tickfontsize   = fsize, 
        legendfontsize = fsize-3, 
        xguidefontsize = fsize, 
        yguidefontsize = fsize, 
        titlefontsize  = fsize, 
        margin = 5Plots.mm,
        bottom_margin = 7Plots.mm,  
        lw     = 3, 
        )

    return x0, str, p, ts, dt

end 