using Plots 
using StatsPlots 
using Statistics 

## ============================================ ##
# plot sindy and gpsindy stats boxplot 

export boxplot_err 
function boxplot_err( noise_vec, sindy_err_vec, gpsindy_err_vec )

    xmin, dx, xmax = min_d_max(noise_vec) 

    p_Ξ = [] 
    for i = 1:2
        ymin, dy, ymax = min_d_max([ sindy_err_vec[:,i] ; gpsindy_err_vec[:,i] ]) 
        p_ξ = scatter( noise_vec, sindy_err_vec[:,i], shape = :circle, ms = 2, c = :blue, label = "SINDy" )
            boxplot!( p_ξ, noise_vec, sindy_err_vec[:,i], bar_width = 0.04, lw = 1, fillalpha = 0.2, c = :blue, linealpha = 0.5 )
            scatter!( p_ξ, noise_vec, gpsindy_err_vec[:,i], shape = :xcross, c = :red, label = "GPSINDy" )
            boxplot!( p_ξ, noise_vec, gpsindy_err_vec[:,i], bar_width = 0.02, lw = 1, fillalpha = 0.2, c = :red, linealpha = 0.5 ) 
            scatter!( p_ξ, 
                legend = false, 
                xlabel = "noise", 
                title  = string( "\n ||ξ", i, "_true - ξ", i, "_discovered||" ), 
                xticks = xmin : dx : xmax, 
                # yticks = ymin : dy : ymax, 
                ) 
        push!(p_Ξ, p_ξ)
    end 
    p = deepcopy(p_Ξ[end])  
    plot!(p, 
        legend     = (-0.2,0.6) , 
        framestyle = :none , 
        title      = "", 
        )
    push!( p_Ξ, p ) 
    p_Ξ = plot(p_Ξ ... , 
        layout = grid( 1, 3, widths = [0.45, 0.45, 0.45] ) , 
        size   = [ 800 300 ], 
        margin = 8Plots.mm,
        ) 
    display(p_Ξ)

    return p_Ξ
end 

## ============================================ ##
# plot prey vs. predator 

using Plots 
using Latexify 

export plot_test_data 
function plot_test_data( t_test, x_test, t_sindy_val, x_sindy_val, t_gpsindy_val, x_gpsindy_val )

    # scalefontsizes(1.1)
    ptitles = ["Prey", "Predator"]
    
    # determine xtick range 
    xmin, dx, xmax = min_d_max( t_test )
    
    n_vars   = size(x_test, 2)
    plot_vec = [] 

    for i = 1 : n_vars 
        
        ymin, dy, ymax = min_d_max( x_test[:,i] ) 

        # display test data 
        p = plot(t_test, x_test[:,i], 
            c      = RGB( 0, 0.35, 1 ) , 
            lw     = 3 ,  
            label  = "test (20%)", 
            xlim   = ( xmin, xmax ) , 
            ylim   = ( ymin - dy/3, ymax + dy/3 ) , 
            xticks = xmin : dx : xmax , 
            yticks = ymin : dy : ymax , 
            xlabel = "Time (s)", 
            title  = string( ptitles[i], ", ", latexify( "x_$(i)" ) ), 
            ) 
        plot!(t_sindy_val, x_sindy_val[:,i], 
            ls    = :dash , 
            # c     = :red , 
            c     = RGB( 1, 0.25, 0 ) , 
            lw    = 3 , 
            label = "SINDy" ,  
            )
        plot!(t_gpsindy_val, x_gpsindy_val[:,i], 
            ls    = :dashdot , 
            c     = RGB(0, 0.75, 0) , 
            lw    = 2 , 
            label = "GP SINDy" ,  
            )
    
        push!( plot_vec, p ) 
    
    end 
    
    p = deepcopy(plot_vec[end])  
        plot!(p, 
        legend     = (-0.1,0.6) , 
        # foreground_color_legend = nothing , 
        framestyle = :none , 
        title      = "", 
        )
    push!( plot_vec, p ) 
    
    p_train_val = plot(plot_vec ... , 
        # layout = (1, n_vars+1), 
        layout = grid( 1, n_vars+1, widths=[0.45, 0.45, 0.45] ) , 
        size   = [ n_vars*400 250 ], 
        margin = 5Plots.mm,
        bottom_margin = 7Plots.mm,  
        # plot_title = "Training vs. Validation Data", 
        # titlefont = font(16), 
        )
    display(p_train_val) 

end 

## ============================================ ##
# plot prey vs. predator 

using Plots 
using Latexify 

export plot_prey_predator 
function plot_prey_predator( t_train, x_train, t_test, x_test, t_sindy_val, x_sindy_val, t_gpsindy_val, x_gpsindy_val )

    # scalefontsizes(1.1)
    ptitles = ["Prey", "Predator"]
    
    # determine xtick range 
    t   = [t_train; t_test] 
    x   = [x_train ; x_test]
    xmin, dx, xmax = min_d_max( t )
    
    n_vars   = size(x_train, 2)
    plot_vec = [] 

    # determine if test data in middle 
    tdiff = diff(vec(t_train))
    ttol  = 1/2*(maximum(t_test) - minimum(t_test)) 
    if any(tdiff .> ttol)
        portion_mid = true 
        ind = findfirst( tdiff .> ttol )
        t_train_B = t_train[ind+1:end,:]
        x_train_B = x_train[ind+1:end,:]
        t_train = t_train[1:ind,:]
        x_train = x_train[1:ind,:] 
    else
        portion_mid = false 
    end

    for i = 1 : n_vars 
        
        ymin, dy, ymax = min_d_max( x[:,i] ) 

        p = plot(t_train, x_train[:,i], 
                c      = :gray, 
                label  = "train (80%)", 
                xlim   = ( xmin, xmax ) , 
                ylim   = ( ymin - dy/3, ymax + dy/3 ) , 
                xticks = xmin : dx : xmax , 
                yticks = ymin : dy : ymax , 
                xlabel = "Time (s)", 
                title  = string( ptitles[i], ", ", latexify( "x_$(i)" ) ), 
                ) 
        # display training data 
        if portion_mid 
            plot!( t_train_B, x_train_B[:,i], 
                c      = :gray, 
                primary = false, 
                ) 
        end 
        plot!(t_test, x_test[:,i], 
            # ls = :dash, 
            # c     = :blue,
            c     = RGB( 0, 0.35, 1 ) , 
            lw    = 3 ,  
            label = "test (20%)" , 
            )
        plot!(t_sindy_val, x_sindy_val[:,i], 
            ls    = :dash , 
            # c     = :red , 
            c     = RGB( 1, 0.25, 0 ) , 
            lw    = 3 , 
            label = "SINDy" ,  
            )
        plot!(t_gpsindy_val, x_gpsindy_val[:,i], 
            ls    = :dashdot , 
            c     = RGB(0, 0.75, 0) , 
            lw    = 2 , 
            label = "GP SINDy" ,  
            )
    
        push!( plot_vec, p ) 
    
    end 
    
    p = deepcopy(plot_vec[end])  
        plot!(p, 
        legend     = (-0.1,0.6) , 
        # foreground_color_legend = nothing , 
        framestyle = :none , 
        title      = "", 
        )
    push!( plot_vec, p ) 
    
    p_train_val = plot(plot_vec ... , 
        # layout = (1, n_vars+1), 
        layout = grid( 1, n_vars+1, widths=[0.45, 0.45, 0.45] ) , 
        size   = [ n_vars*400 250 ], 
        margin = 5Plots.mm,
        bottom_margin = 7Plots.mm,  
        # plot_title = "Training vs. Validation Data", 
        # titlefont = font(16), 
        )
    display(p_train_val) 

end 


## ============================================ ##
# plot derivatives 

using Plots 

export plot_deriv
function plot_deriv(t, dx_true, dx_fd, dx_tv, str) 

    n_vars = size(dx_true, 2) 

    xmin, dx, xmax = min_d_max( t )

    plot_vec_dx = [] 
    for j in 1 : n_vars
        ymin, dy, ymax = min_d_max( dx_true[:,j] ) 
        plt = plot(t, dx_true[:,j], 
            title = "dx $(j)", label = "true", 
            xticks = xmin : dx : xmax , 
            yticks = ymin : dy : ymax , 
            xlabel = "Time (s)"
            ) 
            plot!(t, dx_fd[:,j], ls = :dash, label = "finite diff" )
            # plot!(t, dx_tv[:,j], ls = :dash, label = "var diff" )
    push!( plot_vec_dx, plt ) 
    end

    plot_dx = plot(plot_vec_dx ... , 
        layout = (1, n_vars), 
        size = [n_vars*400 250], 
        # plot_title = "Derivatives. ODE fn = $( str )" 
        ) 
    display(plot_dx) 

    return plot_dx 

end 


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
        layout = (1, n_vars), 
        size = [n_vars*400 250 ], 
        xlabel = "Time (s)", 
        plot_title = "Dynamics. ODE fn = $( str )" ) 
    display(plot_x)      

    return plot_x 

end 


## ============================================ ##

export plot_dx_sindy_gpsindy
function plot_dx_sindy_gpsindy( t, dx_true, dx_noise, Θx_sindy, Ξ_sindy, Θx_gpsindy, Ξ_gpsindy ) 
            
    n_vars = size(dx_true, 2) 
    plt_nvars = [] 
    for i = 1 : n_vars 
        plt = plot( t, dx_true[:,i], label = "true", c = :black ) 
        scatter!( plt, t, dx_noise[:,i], label = "train (noise)", c = :black, ms = 3 ) 
        plot!( plt, t, Θx_sindy * Ξ_sindy[:,i], label = "SINDy", c = :red )   
        plot!( plt, t, Θx_gpsindy * Ξ_gpsindy[:,i], label = "GPSINDy", ls = :dash, c = :cyan )   
        plot!( plt, legend = :outerright, size = [800 300], title = string( "Fitting ξ", i ), xlabel = "Time (s)" ) 
        push!( plt_nvars, plt ) 
    end 
    plt_nvars = plot( plt_nvars ... , 
        layout = (2,1), 
        size   = [800 600] 
        ) 
    display(plt_nvars) 

    return plt_nvars 
end 
