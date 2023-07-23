using GaussianSINDy

## ============================================ ##
# create data 

# constants 
α = 1.0  ; ρ = 1.0   
noise = 0.1  
λ     = 0.1 
abstol = 1e-2 ; reltol = 1e-2  

# choose ODE, plot states --> measurements 
fn = predator_prey 
x0, dt, t, x_true, dx_true, dx_fd, p = ode_states(fn, 0, 2) 

# noise 
x_noise  = x_true + noise*randn( size(x_true, 1), size(x_true, 2) )
dx_noise = fdiff(t, x_noise, 2)
# dx_noise = dx_true + noise*randn( size(x_true, 1), size(x_true, 2) )

# truth coeffs 
n_vars = size(x_true, 2) ; poly_order = n_vars 
Ξ_true = SINDy_test( x_true, dx_true, λ ) 

# standardize true data 
x_stand_true  = stand_data( t, x_true )    
# dx_stand_true = dx_true_fn( t, x_stand_true, p, fn )  
dx_stand_true = stand_data( t, dx_true ) 

# add noise to standardized data 
x_stand_noise = x_stand_true + noise*randn( size(x_true, 1), size(x_true, 2) ) 
dx_stand_noise = dx_stand_true + noise*randn( size(x_true, 1), size(x_true, 2) ) 

## ============================================ ##
# STANDARDIZED: try various SINDy's 

Ξ_stand_true  = SINDy_test( x_stand_true, dx_stand_true, λ ) 
Ξ_stand_sindy = SINDy_test( x_stand_noise, dx_stand_noise, λ ) 

x_stand_GP   = post_dist_GP( t, t, x_stand_noise ) 
dx_stand_GP  = post_dist_GP( t, t, dx_stand_noise ) 

Ξ_stand_sindy_GP = SINDy_test( x_stand_GP, dx_stand_GP, λ ) 

# ----------------------- #
# using GPSINDy 

# placeholder 
Θx = pool_data_test( x_stand_GP, n_vars, poly_order ) 
Ξ_gpsindy_GP, hist_nvars = gpsindy( t, dx_stand_GP, Θx, 2*λ, α, ρ, abstol, reltol )  

# placeholder 
Θx = pool_data_test( x_stand_noise, n_vars, poly_order ) 
Ξ_gpsindy, hist_nvars = gpsindy( t, dx_stand_noise, Θx, 2*λ, α, ρ, abstol, reltol )  

println( "(stand) true - SINDy err    = ", norm( Ξ_stand_true - Ξ_stand_sindy )  ) 
println( "(stand) true - SINDy_GP err = ", norm( Ξ_stand_true - Ξ_stand_sindy_GP )  ) 
println( "(stand) true - GPSINDy err  = ", norm( Ξ_stand_true - Ξ_gpsindy_GP )  ) 
println( "(stand) true - GPSINDy err  = ", norm( Ξ_stand_true - Ξ_gpsindy )  ) 

## ============================================ ##
# NON-STANDARDIZED: try various SINDy's 

Ξ_true  = SINDy_test( x_true, dx_true, λ ) 
Ξ_sindy = SINDy_test( x_noise, dx_noise, λ ) 

x_GP   = post_dist_GP( t, t, x_noise ) 
dx_GP  = post_dist_GP( t, t, dx_noise ) 

Ξ_sindy_GP = SINDy_test( x_GP, dx_GP, λ ) 

# ----------------------- #
# using GPSINDy 

# placeholder 
Θx = pool_data_test( x_GP, n_vars, poly_order ) 
Ξ_gpsindy_GP, hist_nvars = gpsindy( t, dx_GP, Θx, 3*λ, α, ρ, abstol, reltol )  

# placeholder 
Θx = pool_data_test( x_noise, n_vars, poly_order ) 
Ξ_gpsindy, hist_nvars = gpsindy( t, dx_noise, Θx, 3*λ, α, ρ, abstol, reltol )  

println( "true - SINDy err    = ", norm( Ξ_true - Ξ_sindy )  ) 
println( "true - SINDy_GP err = ", norm( Ξ_true - Ξ_sindy_GP )  ) 
println( "true - GPSINDy err  = ", norm( Ξ_true - Ξ_gpsindy_GP )  ) 
println( "true - GPSINDy err  = ", norm( Ξ_true - Ξ_gpsindy )  ) 

## ============================================ ##
# use GP to smooth data 


p_states = [] 
for i = 1:n_vars 
    plt = plot( t, x_stand_true[:,i], label = "true", c = :green )
    scatter!( plt, t, x_stand_noise[:,i], label = "noise", c = :black, ms = 3 )
    scatter!( plt, t, x_stand_smooth[:,i], label = "smooth", c = :red, ms = 1, markerstrokewidth = 0 )
    plot!( plt, legend = :outerright, title = string( "state ", i ) )    
    push!(p_states, plt)
end 
p_states = plot(p_states ... ,   
    layout = (2,1), 
    size   = [800 300]
) 
display(p_states) 
