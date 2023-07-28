using GaussianSINDy
using GaussianProcesses
using Plots 
using Optim 

# choose ODE, plot states --> measurements 
fn = predator_prey 
x0, dt, t, x_true, dx_true, dx_fd, p = ode_states(fn, 0, 2) 

noise = 0.2 

x_stand  = stand_data( t, x_true ) 
dx_stand = dx_true_fn( t, x_true, p, fn ) 
x_noise  = x_stand + noise*randn( size(x_true, 1), size(x_true, 2) ) 
dx_noise = dx_stand + noise*randn( size(x_true, 1), size(x_true, 2) ) 
x_stand = x_stand[:,1] ; x_noise = x_noise[:,1] ; x_true = x_true[:,1] 

x_train = t 
x_test  = collect( t[1] : 0.1 : t[end] ) 
# x_test  = x_train 
y_train = x_noise


## ============================================ ##
# hp optimization (toolbox) 

function def_plt(t, dx_stand, y_train)   

    plt  = plot( title = "Gaussian Process (opt HPs)", legend = :outerright, size = [800 300], ylim = ( -5, 5 ) ) 
    plot!( plt, t, dx_stand, label = "true", c = :black ) 
    scatter!( plt, t, y_train, label = "train (noise)", c = :black, ms = 3 ) 

    return plt 
end 

a = Animation() 

plt = def_plt( t, x_stand, y_train ) 
frame(a, plt) 

μ_SE, Σ_SE, hp = post_dist_SE( x_train, x_test, y_train ) 
    plt = def_plt( t, x_stand, y_train ) 
    plot!( plt, x_test, μ_SE, label = "SE", ribbon = Σ_SE ) 
frame(a, plt) 

μ_M12A, Σ_M12A, hp = post_dist_M12A( x_train, x_test, y_train ) 
    plt = def_plt( t, x_stand, y_train ) 
    plot!( plt, x_test, μ_M12A, label = "M12A", ribbon = Σ_M12A ) 
frame(a, plt) 

μ_M32A, Σ_M32A, hp = post_dist_M32A( x_train, x_test, y_train ) 
    plt = def_plt( t, x_stand, y_train ) 
    plot!( plt, x_test, μ_M32A, label = "M32A", ribbon = Σ_M32A ) 
frame(a, plt) 

μ_M52A, Σ_M52A, hp = post_dist_M52A( x_train, x_test, y_train ) 
    plt = def_plt( t, x_stand, y_train ) 
    plot!( plt, x_test, μ_M52A, label = "M52A", ribbon = Σ_M52A ) 
frame(a, plt) 
# println( "dx_true - μ_M52A = ", norm( dx_true - μ_M52A ) ) 

μ_M12I, Σ_M12I, hp = post_dist_M12I( x_train, x_test, y_train ) 
    plt = def_plt( t, x_stand, y_train ) 
    plot!( plt, x_test, μ_M12I, label = "M12I", ribbon = Σ_M12I ) 
frame(a, plt) 
# println( "dx_true - μ_M12I = ", norm( dx_true - μ_M12I ) ) 

μ_M32I, Σ_M32I, hp = post_dist_M32I( x_train, x_test, y_train ) 
    plt = def_plt( t, x_stand, y_train ) 
    plot!( plt, x_test, μ_M32I, label = "M32I", ribbon = Σ_M32I ) 
frame(a, plt) 
# println( "dx_true - μ_M32I = ", norm( dx_true - μ_M32I ) ) 

μ_M52I, Σ_M52I, hp = post_dist_M52I( x_train, x_test, y_train ) 
    plt = def_plt( t, x_stand, y_train ) 
    plot!( plt, x_test, μ_M52I, label = "M52I", ribbon = Σ_M52I ) 
frame(a, plt) 

# μ_per, Σ_per, hp = post_dist_per( x_train, x_test, y_train ) 
#     plt = def_plt( t, dx_stand, y_train ) 
#     plot!( plt, x_test, μ_per, label = "Per", ls = :dashdot, ribbon = ( μ_per - Σ_per, μ_per + Σ_per )  ) 
# frame(a, plt) 
# println( "dx_true - μ_M52I = ", norm( dx_true - μ_M52I ) ) 

# μ_post, Σ_post, hp_post = post_dist_hp_opt( x_train, y_train, x_test )
# σ²_manual = diag( Σ_post ) 
#     plot!( plt, x_test, μ_post, label = "manual", c = :cyan, ls = :dashdot, ribbon = ( μ_post - σ²_manual, μ_post + σ²_manual )  )

g = gif(a, fps = 1.0) 
display(g) 
display(plt) 

## ============================================ ## 

# choose ODE, plot states --> measurements 
fn = predator_prey 
x0, dt, t, x, dx_true, dx_fd = ode_states(fn, 0, 2) 

dx_noise = 0.2 

dx_stand = stand_data(t, dx_true) 
dx_noise = dx_stand + dx_noise*randn( size(dx_true, 1), size(dx_true, 2) ) 

x_train = t 
x_test  = collect( t[1] : 0.1 : t[end] ) 
# x_test  = x_train 
y_train = dx_noise

# kernel  
mZero     = MeanZero() ;            # zero mean function 
kern      = Mat32Iso( 0.0, 0.0 ) ;        # squared eponential kernel (hyperparams on log scale) 
log_noise = log(0.1) ;              # (optional) log std dev of obs noise 

n_vars = size(x, 2) 
    
# loop through states 
y_smooth = zeros( length(x_test), size(y_train, 2) ) 
Σ        = 0 * y_smooth 
for i = 1:n_vars 

    # fit GP 
    gp      = GP(x_train, y_train[:,i], mZero, kern, log_noise) 
    optimize!(gp) 
    μ, σ²   = predict_y( gp, x_test )    

    y_smooth[:,i] = μ 
    Σ[:,i]        = σ²

end 

y_test, Σ_test, hp_test = post_dist_M32I( x_train, x_test, y_train ) 


## ============================================ ##

x_smooth, Σ_xsmooth, hp = post_dist_SE( t, t, x_noise )  

# y_train = x_stand 
# y_train = x_noise 
y_train = x_smooth 

μ, Σ = post_dist_SE( x_smooth, x_smooth, dx_noise )  
μ_man, Σ_man = post_dist( y_vec, dx_noise, y_vec, σ_f, l, σ_n ) 
# μ_man, Σ_man, hp = post_dist_hp_opt( y_vec, dx_noise[:,1], y_vec )  

i = 2 
plot( legend = :outerright, xlabel = "Time (s)", title = "dx = f(x)" ) 
plot!( t, dx_true[:,i], label = "true", legend = :outerright )
plot!( t, dx_noise[:,i], ls = :dash, label = "noise" )
plot!( t, μ[:,i], ls = :dashdot, label = "GP" )   


## ============================================ ##


μ_SE, Σ_SE, hp = post_dist_SE( x_train, x_test, y_vec ) 
    plt = def_plt( t, x_stand, y_train ) 
    plot!( plt, x_test, μ_SE, label = "SE", ribbon = ( μ_SE - Σ_SE, μ_SE + Σ_SE )  ) 



