
## ============================================ ##
# plot state 

using Plots 

export plot_dyn 
function plot_dyn(t, x, str)

    n_vars = size(x,2) 

    # construct empty vector for plots 
    plot_vec_x = [] 
    for i = 1:n_vars 
        plt = plot(t, x[:,i], title = "State $(i)")
        push!(plot_vec_x, plt)
    end 
    plot_x = plot(plot_vec_x ..., 
        layout = (n_vars,1), 
        size = [600 n_vars*300], 
        xlabel = "Time (s)", 
        plot_title = "Dynamics. ODE fn = $( str )" ) 
    display(plot_x)      

    return plot_x 

end 

## ============================================ ##
# plot derivatives 

using Plots 

export plot_deriv
function plot_deriv(t, dx_true, dx_fd, str) 

    n_vars = size(dx_true, 2) 

    plot_vec_dx = [] 
    for j in 1 : n_vars
        plt = plot(t, dx_true[:,j], 
            title = "State $(j)", label = "true" ) 
            plot!(t, dx_fd[:,j], ls = :dash, label = "finite diff" )
    push!( plot_vec_dx, plt ) 
    end

    plot_dx = plot(plot_vec_dx ... , 
        layout = (n_vars, 1), 
        size = [600 n_vars*300], 
        plot_title = "Derivatives. ODE fn = $( str )" )
    display(plot_dx) 

    return plot_dx 

end 