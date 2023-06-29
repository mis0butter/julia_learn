

## ============================================ ##
# split into training and validation data 

export split_train_test 
function split_train_test(x, test_fraction, portion)

    # if test data = LAST portion 
    if portion == 1/test_fraction 

        ind = Int(round( size(x,1) * (1 - test_fraction) ))  

        x_train = x[1:ind,:]
        x_test  = x[ind:end,:] 

    # if test data = FIRST portion 
    elseif portion == 1 

        ind = Int(round( size(x,1) * (test_fraction) ))  

        x_train = x[ind:end,:] 
        x_test  = x[1:ind,:]

    # test data is in MIDDLE portion 
    else 

        ind1 = Int(round( size(x,1) * (test_fraction*( portion-1 )) )) 
        ind2 = Int(round( size(x,1) * (test_fraction*( portion )) )) 

        x_test  = x[ ind1:ind2,: ]
        x_train = [ x[ 1:ind1,: ] ; x[ ind2:end,: ] ]

    end 

    return x_train, x_test 

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

export min_max_d 
function min_d_max( x )

    xmin = round( minimum(x), digits = 1 )  
    xmax = round( maximum(x), digits = 1 ) 
    dx   = round( ( xmax - xmin ) / 2, digits = 1 ) 

    return xmin, dx, xmax  

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
# plot prey vs. predator 

using Plots 
using Latexify 

export plot_prey_predator 
function plot_prey_predator( t_train, x_train, t_test, x_test, t_sindy_val, x_sindy_val, t_gpsindy_val, x_gpsindy_val )

    # scalefontsizes(1.1)
    ptitles = ["Prey", "Predator"]
    
    # determine xtick range 
    t_i = Int(round(3/4*length(t_train))) 
    t   = [t_train[t_i : end] ; t_test]
    x   = [x_train ; x_test]
    xmin, dx, xmax = min_d_max( t )
    
    n_vars   = size(x_train, 2)
    plot_vec = [] 
    for i = 1 : n_vars 
        
        ymin, dy, ymax = min_d_max( x[:,i] ) 

        # display training data 
        p = plot(t_train, x_train[:,i], 
            c      = :gray, 
            label  = "train (70%)", 
            xlim   = ( xmin, xmax ) , 
            ylim   = ( ymin - dy/3, ymax + dy/3 ) , 
            xticks = xmin : dx : xmax , 
            yticks = ymin : dy : ymax , 
            xlabel = "Time (s)", 
            title  = string( ptitles[i], ", ", latexify( "x_$(i)" ) ), 
            ) 
        plot!(t_test, x_test[:,i], 
            # ls = :dash, 
            # c     = :blue,
            c     = RGB( 0, 0.35, 1 ) , 
            lw    = 3 ,  
            label = "test (30%)" , 
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

