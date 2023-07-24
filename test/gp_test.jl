using GaussianSINDy
using GaussianProcesses
using Plots 
using Optim 

# choose ODE, plot states --> measurements 
fn = predator_prey 
x0, dt, t, x, dx_true, dx_fd = ode_states(fn, 0, 2) 

dx_noise = 0.2 

dx_stand = stand_data(t, dx_true) 
dx_noise = dx_stand + dx_noise*randn( size(dx_true, 1), size(dx_true, 2) ) 
dx_stand = dx_stand[:,1] ; dx_noise = dx_noise[:,1] ; dx_true = dx_true[:,1] 

x_train = t 
x_test  = collect( t[1] : 0.1 : t[end] ) 
# x_test  = x_train 
y_train = dx_noise


## ============================================ ##
# hp optimization (toolbox) 

function def_plt(t, dx_stand, y_train)   

    plt = plot( t, dx_stand, label = "true", c = :black, title = "Gaussian Process (opt HPs)" ) 
    scatter!( plt, t, y_train, label = "train (noise)", c = :red, ms = 3 ) 
    plot!( plt, legend = :outerright, size = [800 300], ylim = ( -5, 5 ) ) 

    return plt 
end 

a = Animation() 

plt = def_plt( t, dx_stand, y_train ) 
frame(a, plt) 

μ_SE, Σ_SE, hp = post_dist_SE( x_train, x_test, y_train ) 
    plt = def_plt( t, dx_stand, y_train ) 
    plot!( plt, x_test, μ_SE, label = "SE", ls = :dash, ribbon = ( μ_SE - Σ_SE, μ_SE + Σ_SE )  ) 
frame(a, plt) 

μ_M12A, Σ_M12A, hp = post_dist_M12A( x_train, x_test, y_train ) 
    plt = def_plt( t, dx_stand, y_train ) 
    plot!( plt, x_test, μ_M12A, label = "M12A", ls = :dashdot, ribbon = ( μ_M12A - Σ_M12A, μ_M12A + Σ_M12A )  ) 
frame(a, plt) 

μ_M32A, Σ_M32A, hp = post_dist_M32A( x_train, x_test, y_train ) 
    plt = def_plt( t, dx_stand, y_train ) 
    plot!( plt, x_test, μ_M32A, label = "M32A", ls = :dashdot, ribbon = ( μ_M32A - Σ_M32A, μ_M32A + Σ_M32A )  ) 
frame(a, plt) 

μ_M52A, Σ_M52A, hp = post_dist_M52A( x_train, x_test, y_train ) 
    plt = def_plt( t, dx_stand, y_train ) 
    plot!( plt, x_test, μ_M52A, label = "M52A", ls = :dashdot, ribbon = ( μ_M52A - Σ_M52A, μ_M52A + Σ_M52A )  ) 
frame(a, plt) 
# println( "dx_true - μ_M52A = ", norm( dx_true - μ_M52A ) ) 

μ_M12I, Σ_M12I, hp = post_dist_M12I( x_train, x_test, y_train ) 
    plt = def_plt( t, dx_stand, y_train ) 
    plot!( plt, x_test, μ_M12I, label = "M12I", ls = :dashdot, ribbon = ( μ_M12I - Σ_M12I, μ_M12I + Σ_M12I )  ) 
frame(a, plt) 
# println( "dx_true - μ_M12I = ", norm( dx_true - μ_M12I ) ) 

μ_M32I, Σ_M32I, hp = post_dist_M32I( x_train, x_test, y_train ) 
    plt = def_plt( t, dx_stand, y_train ) 
    plot!( plt, x_test, μ_M32I, label = "M32I", ls = :dashdot, ribbon = ( μ_M32I - Σ_M32I, μ_M32I + Σ_M32I )  ) 
frame(a, plt) 
# println( "dx_true - μ_M32I = ", norm( dx_true - μ_M32I ) ) 

μ_M52I, Σ_M52I, hp = post_dist_M52I( x_train, x_test, y_train ) 
    plt = def_plt( t, dx_stand, y_train ) 
    plot!( plt, x_test, μ_M52I, label = "M52I", ls = :dashdot, ribbon = ( μ_M52I - Σ_M52I, μ_M52I + Σ_M52I )  ) 
frame(a, plt) 

μ_per, Σ_per, hp = post_dist_per( x_train, x_test, y_train ) 
    plt = def_plt( t, dx_stand, y_train ) 
    plot!( plt, x_test, μ_per, label = "Per", ls = :dashdot, ribbon = ( μ_per - Σ_per, μ_per + Σ_per )  ) 
frame(a, plt) 
# println( "dx_true - μ_M52I = ", norm( dx_true - μ_M52I ) ) 

# μ_post, Σ_post, hp_post = post_dist_hp_opt( x_train, y_train, x_test )
# σ²_manual = diag( Σ_post ) 
#     plot!( plt, x_test, μ_post, label = "manual", c = :cyan, ls = :dashdot, ribbon = ( μ_post - σ²_manual, μ_post + σ²_manual )  )

g = gif(a, fps = 1.0) 
display(g) 
display(plt) 

## ============================================ ## 

for i = 1:3 
    if hp[i] == 0 
        hp[i] = 1e-5 
    end 
end 

# kernel  
mZero     = MeanZero() ;            # zero mean function 
kern      = SE( log(hp[2]), log(hp[1]) ) ;        # squared eponential kernel (hyperparams on log scale) 
log_noise = log(hp[3]) ;              # (optional) log std dev of obs noise 

# fit GP 
# y_train = dx_train - Θx*ξ   
t_train = t 
y_train = dx_noise[:,1]
gp      = GP(t_train, y_train, mZero, kern, log_noise) 

μ_opt, σ²_opt = predict_y( gp, x_test )
scatter!( x_test, μ_opt, ms = 2 )

