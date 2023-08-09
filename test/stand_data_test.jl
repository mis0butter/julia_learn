using GaussianSINDy
using LineSearches 


## ============================================ ##
# create data 

# constants 
α      = 1.0  ; ρ = 1.0   
noise  = 0.1 
λ      = 0.1 
abstol = 1e-3 ; reltol = 1e-3 

# choose ODE, plot states --> measurements 
fn = pendulum 
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


## ============================================ ##

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
Ξ_gpsindy_post  = SINDy_test( x_GP, dx_post, λ ) 

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
