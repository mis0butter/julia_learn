using GaussianSINDy
using GaussianProcesses
using Plots 
using Optim 

# choose ODE, plot states --> measurements 
fn = predator_prey 
x0, dt, t, x, dx_true, dx_fd = ode_states(fn, 0, 2) 

dx_noise = 0.1 

dx_fd = dx_true + dx_noise*randn( size(dx_true, 1), size(dx_true, 2) ) 
dx_fd = dx_fd[:,1] ; dx_true = dx_true[:,1] 

# dx_fd = sin.(t) + 0.05*randn(length(t));   #regressors

## ============================================ ## 

σ_f = log(1.0) ; l = log(1.0) ; σ_n = log(0.1) 

# kernel  
mZero     = MeanZero() ;            # zero mean function 
kern      = SE( 0.0, 0.0 ) ;        # squared eponential kernel (hyperparams on log scale) 
log_noise = log(0.1) ;              # (optional) log std dev of obs noise 

# fit GP 
# y_train = dx_train - Θx*ξ   
x_train = t 
y_train = dx_fd
gp      = GP(x_train, y_train, mZero, kern, log_noise) 

# tests 
x_test  = t 
μ, σ²   = predict_y( gp, x_test )
μ_post, Σ_post = post_dist( x_train, y_train, x_test, exp(σ_f), exp(l), exp(σ_n) ) 

p_gp = plot(gp; xlabel="x", ylabel="y", title="Gaussian Process", legend = true, fmt=:png) 
plot!( x_test, μ, label = "gp toolbox", c = :red )
plot!( t, dx_true, label = "true", c = :green ) 
plot!( x_test, μ_post, label = "post", ls = :dash, c = :cyan, lw = 1.5 )

## ============================================ ##
# hp optimization (toolbox) 

# toolbox 
optimize!(gp) 
plot!( gp, title = "Opt HPs", label = "opt toolbox", legend = true ) 
plot!( p_gp, t, dx_true, label = "true", c = :green ) 

## ============================================ ##
# hp optimization (June) --> post mean  

μ_post, Σ_post, hp_post = post_dist_hp_opt( x_train, y_train, x_test )

plot!( p_opt, x_test, μ_post, label = "post"  )
# plot!( p_opt, x_test, μ_post2, label = "post2", ls = :dash )

## ============================================ ##
# optimize hps 

result  = optimize!(gp) 

σ_f = result.minimizer[1] 
l   = result.minimizer[2] 
σ_n = result.minimizer[3] 
hp  = [σ_f, l, σ_n] 

## ============================================ ##

# kernel  
mZero     = MeanZero() ;            # zero mean function 
kern      = SE( σ_f, l) ;        # squared eponential kernel (hyperparams on log scale) 
log_noise = log(σ_n) ;              # (optional) log std dev of obs noise 

# fit GP 
# y_train = dx_train - Θx*ξ   
t_train = t 
y_train = dx_fd[:,1]
gp      = GP(t_train, y_train, mZero, kern, log_noise) 

μ_opt, σ²_opt = predict_y( gp, x_test )
scatter!( x_test, μ_opt, ms = 2 )

