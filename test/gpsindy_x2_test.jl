using GaussianSINDy
# using LineSearches 


## ============================================ ##

# choose ODE, plot states --> measurements 
fn = predator_prey 
# constants 
λ  = 0.1 

# # set up noise vec 
# noise_vec = [] 
# noise_vec_iter = 0.0 : 0.01 : 0.3  
# for i in noise_vec_iter 
#     for j = 1:10 
#         push!(noise_vec, i)
#     end 
# end 
# noise_vec = collect( 0 : 0.05 : 0.2 ) 
# noise_vec = [ 0.1 ]   
noise = 0.0 

# ----------------------- #
# start MC loop 

Ξ_vec = [] 
Ξ_hist = Ξ_struct( [], [], [], [] ) 
Ξ_err_hist = Ξ_err_struct( [], [], [] ) 
# for noise = noise_vec 
#     Ξ_hist, Ξ_err_hist = gpsindy_x2( fn, noise, λ, Ξ_hist, Ξ_err_hist ) 
# end 


# ----------------------- #

# generate true states 
x0, dt, t, x_true, dx_true, dx_fd, p = ode_states(fn, 0, 2) 

# truth coeffs 
n_vars = size(x_true, 2) ; poly_order = n_vars 
Ξ_true = SINDy_test( x_true, dx_true, λ ) 

# add noise 
println( "noise = ", noise ) 
x_noise  = x_true + noise*randn( size(x_true, 1), size(x_true, 2) )
dx_noise = dx_true + noise*randn( size(dx_true, 1), size(dx_true, 2) )

# split into training and test data 
test_fraction = 0.2 
portion       = 5 
t_train, t_test   = split_train_test(t, test_fraction, portion) 
x_train, x_test   = split_train_test(x_noise, test_fraction, portion) 
dx_train, dx_test = split_train_test(dx_noise, test_fraction, portion) 
dx_true_train, dx_true_test = split_train_test(dx_true, test_fraction, portion) 

# ----------------------- #
# standardize  
x_train_stand  = stand_data( t_train, x_train ) 
dx_train_stand = fdiff( t_train, x_train_stand, 2 ) 
# dx_stand = stand_data( t, dx_true ) 
# dx_stand = dx_true_fn( t, x_stand, p, fn ) 

## ============================================ ##



# ----------------------- #
# SINDy by itself 

Θx_sindy = pool_data_test( x_train_stand, n_vars, poly_order ) 
Ξ_sindy  = SINDy_test( x_train_stand, dx_train_stand, λ ) 

# ----------------------- #
# GPSINDy (first) 

# step -1 : smooth x measurements with t (temporal)  
x_GP, Σ_xsmooth, hp   = post_dist_SE( t_train, x_train_stand, t_train )  

# step 0 : smooth dx measurements with x_GP (non-temporal) 
dx_GP, Σ_dxsmooth, hp = post_dist_SE( x_GP, dx_train_stand, x_GP )  

# SINDy 
Θx_gpsindy = pool_data_test(x_GP, n_vars, poly_order) 
Ξ_gpsindy  = SINDy_test( x_GP, dx_GP, λ ) 

# ----------------------- #
# GPSINDy (second) 

# step 2: GP 
dx_mean = Θx_gpsindy * Ξ_gpsindy 
dx_post = gp_post( x_GP, dx_mean, x_GP, dx_train_stand, dx_mean ) 

# step 3: SINDy 
Θx_gpsindy   = pool_data_test( x_GP, n_vars, poly_order ) 
Ξ_gpsindy_x2 = SINDy_test( x_GP, dx_post, λ ) 


## ============================================ ##
# validate data 

using DifferentialEquations

dx_sindy_fn      = build_dx_fn(poly_order, Ξ_sindy) 
dx_gpsindy_fn    = build_dx_fn(poly_order, Ξ_gpsindy) 
dx_gpsindy_x2_fn = build_dx_fn(poly_order, Ξ_gpsindy_x2) 


n_vars = size(x_test, 2) 
x0     = [ x_test[1] ] 
if n_vars > 1 
    x0 = x_test[1,:] 
end 

# dt    = t_test[2] - t_test[1] 
tspan = (t_test[1], t_test[end]) 
prob  = ODEProblem(fn, x0, tspan, p) 

# solve the ODE
sol   = solve(prob, saveat = dt)

x_validate = sol.u ; 
x_validate = mapreduce(permutedims, vcat, x_validate) 
t_validate = sol.t 

## ============================================ ##


t_sindy_val,      x_sindy_val      = validate_data(t_test, x_test, dx_sindy_fn, dt) 
# t_sindy_val,      x_sindy_val      = validate_data(t_test, x_test, fn, dt) 
t_gpsindy_val,    x_gpsindy_val    = validate_data(t_test, x_test, dx_gpsindy_fn, dt) 
t_gpsindy_x2_val, x_gpsindy_x2_val = validate_data(t_test, x_test, dx_gpsindy_x2_fn, dt) 

# plot!! 
plot_states( t_train, x_train, t_test, x_test, t_sindy_val, x_sindy_val, t_gpsindy_val, x_gpsindy_val, t_gpsindy_x2_val, x_gpsindy_x2_val ) 
plot_test_data( t_test, x_test, t_sindy_val, x_sindy_val, t_gpsindy_val, x_gpsindy_val, t_gpsindy_x2_val, x_gpsindy_x2_val ) 


## ============================================ ##
# plot 

Ξ_sindy_err = Ξ_err_hist.sindy ; 
Ξ_gpsindy_err = Ξ_err_hist.gpsindy ; 
Ξ_gpsindy_x2_err = Ξ_err_hist.gpsindy_x2 ; 
plot_med_quarts_gpsindy_x2( Ξ_sindy_err, Ξ_gpsindy_err, Ξ_gpsindy_x2_err, noise_vec ) 


## ============================================ ##
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
Ξ_true   = SINDy_test( x_true, dx_true, λ ) 

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
