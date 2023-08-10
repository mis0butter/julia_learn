using GaussianSINDy
using LineSearches 


## ============================================ ##

# choose ODE, plot states --> measurements 
fn = predator_prey 
# constants 
λ  = 0.1 

# set up noise vec 
# noise_vec = [] 
# noise_vec_iter = 0.0 : 0.01 : 0.1 
# for i in noise_vec_iter 
#     for j = 1:10 
#         push!(noise_vec, i)
#     end 
# end 
# noise_vec = collect( 0 : 0.05 : 0.2 ) 
noise_vec = 0.1  

# ----------------------- #
# start MC loop 

Ξ_vec = [] 
Ξ_hist = Ξ_struct( [], [], [], [] ) 
Ξ_err_hist = Ξ_err_struct( [], [], [] ) 
for noise = noise_vec 
    Ξ_hist, Ξ_err_hist = gpsindy_x2( fn, noise, λ, Ξ_hist, Ξ_err_hist ) 
    # push!( Ξ_true_vec, Ξ_true )
    # push!( Ξ_sindy_err, norm( Ξ_true - Ξ_sindy ) )
    # push!( Ξ_gpsindy_err, norm( Ξ_true - Ξ_gpsindy ) )
    # push!( Ξ_gpsindy_gpsindy_err, norm( Ξ_true - Ξ_gpsindy_gpsindy ) )
    # push!( Ξ_true_vec, Ξ_true )
    # push!( Ξ_sindy_err, norm( Ξ_true[:,2] - Ξ_sindy[:,2] ) )
    # push!( Ξ_gpsindy_err, norm( Ξ_true[:,2] - Ξ_gpsindy[:,2] ) )
    # push!( Ξ_gpsindy_gpsindy_err, norm( Ξ_true[:,2] - Ξ_gpsindy_gpsindy[:,2] ) )
end 

## ============================================ ##
# plot 

plot_med_quarts_gpsindy_gpsindy( Ξ_sindy_err, Ξ_gpsindy_err, Ξ_gpsindy_gpsindy_err, noise_vec ) 










## ============================================ ##



x0, dt, t, x_true, dx_true, dx_fd, p = ode_states(fn, 0, 2) 
    
# truth coeffs 
n_vars = size(x_true, 2) ; poly_order = n_vars 
Ξ_true = SINDy_test( x_true, dx_true, λ ) 
            
## ============================================ ##
# CASE 7

# add noise 
println( "noise = ", noise ) 
x_true   = stand_data( t, x_true ) 
dx_true  = dx_true_fn( t, x_true, p, fn ) 
Ξ_true = SINDy_test( x_true, dx_true, λ ) 

x_noise  = x_true + noise*randn( size(x_true, 1), size(x_true, 2) )
dx_noise = dx_true + noise*randn( size(dx_true, 1), size(dx_true, 2) )

Θx_sindy = pool_data_test( x_noise, n_vars, poly_order ) 
Ξ_sindy  = SINDy_test( x_noise, dx_noise, λ ) 

# smooth measurements 
x_GP, Σ_xsmooth, hp   = post_dist_SE( t, x_noise, t )  
dx_GP, Σ_dxsmooth, hp = post_dist_SE( x_GP, dx_noise, x_GP )  

Θx_gpsindy = pool_data_test(x_GP, n_vars, poly_order) 
Ξ_gpsindy  = SINDy_test( x_GP, dx_GP, λ ) 

# ----------------------- #

# step 2 
dx_mean = Θx_gpsindy * Ξ_gpsindy 

x_vec = [] 
for i = 1 : size(x_noise, 1) 
    push!( x_vec, x_noise[i,:] ) 
end 
dx_post = 0 * dx_noise 

# optimize hyperparameters 
# i = 1 
for i = 1 : size(x_true, 2) 

    # kernel  
    mZero     = MeanZero() ;            # zero mean function 
    kern      = SE( 0.0, 0.0 ) ;        # squared eponential kernel (hyperparams on log scale) 
    log_noise = log(0.1) ;              # (optional) log std dev of obs 
    
    y_train = dx_noise[:,i] - dx_mean[:,i]
    gp      = GP( x_GP', y_train, mZero, kern, log_noise ) 
    
    optimize!( gp, method = LBFGS(linesearch=LineSearches.BackTracking()) ) 

    # return HPs 
    σ_f = sqrt( gp.kernel.σ2 ) ; l = sqrt.( gp.kernel.ℓ2 ) ; σ_n = exp( gp.logNoise.value )  
    hp  = [σ_f, l, σ_n] 

    K = k_SE( σ_f, l, x_vec, x_vec ) 
    dx_post[:,i] = dx_mean[:,i] + K * ( ( K + σ_n^2 * I ) \ ( dx_noise[:,i] - dx_mean[:,i] ) ) 

end 

Θx_gpsindy = pool_data_test(x_GP, n_vars, poly_order) 
Ξ_gpsindy_gpsindy  = SINDy_test( x_GP, dx_post, λ ) 

println( "Ξ_true - Ξ_sindy = ", norm( Ξ_true - Ξ_sindy ) ) 
println( "Ξ_true - Ξ_gpsindy = ", norm( Ξ_true - Ξ_gpsindy ) ) 
println( "Ξ_true - Ξ_gpsindy_gpsindy = ", norm( Ξ_true - Ξ_gpsindy_gpsindy ) ) 


## ============================================ ##
# plot 

p_states = [] 
for i = 1:n_vars 
    plt = plot( t, x_true[:,i], label = "true", c = :green )
    scatter!( plt, x_noise, t[:,i], label = "noise", c = :black, ms = 3 )
    scatter!( plt, t, x_GP[:,i], label = "smooth", c = :red, ms = 1, markerstrokewidth = 0 )
    plot!( plt, legend = :outerright, title = string( "state ", i ) )    
    push!(p_states, plt)
end 
p_states = plot(p_states ... ,   
    layout = (2,1), 
    size   = [800 300]
) 
display(p_states) 
