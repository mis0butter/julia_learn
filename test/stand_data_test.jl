using GaussianSINDy

## ============================================ ##
# create data 

# constants 
α = 1.0  ; ρ = 1.0   
noise = 0.2 
λ     = 0.1 
abstol = 1e-3 ; reltol = 1e-3 

# choose ODE, plot states --> measurements 
fn = pendulum 
x0, dt, t, x_true, dx_true, dx_fd, p = ode_states(fn, 0, 2) 

# noise 
x_noise  = x_true + noise*randn( size(x_true, 1), size(x_true, 2) )
dx_noise = fdiff(t, x_noise, 2)
# dx_noise = dx_true + noise*randn( size(x_true, 1), size(x_true, 2) )

# truth coeffs 
n_vars = size(x_true, 2) ; poly_order = n_vars 
Ξ_true = SINDy_test( x_true, dx_true, λ ) 

# standardize true data 
x_stand_true   = stand_data( t, x_true )    
# dx_stand_true  = dx_true_fn( t, x_stand_true, p, fn )  
dx_stand_true  = stand_data( t, dx_true ) 

# add noise to standardized data 
x_stand_noise  = x_stand_true + noise*randn( size(x_true, 1), size(x_true, 2) ) 
dx_stand_noise = dx_stand_true + noise*randn( size(x_true, 1), size(x_true, 2) ) 

## ============================================ ##
# STANDARDIZED: try various SINDy's 

Ξ_stand_true  = SINDy_test( x_stand_true, dx_stand_true, λ ) 
Ξ_stand_sindy = SINDy_test( x_stand_noise, dx_stand_noise, λ ) 

# ----------------------- #
# using GPSINDy 

x_stand_GP  = post_dist_SE( t, t, x_stand_noise ) 
dx_stand_GP = post_dist_SE( t, t, dx_stand_noise ) 

# placeholder 
Θx = pool_data_test( x_stand_noise, n_vars, poly_order ) 
Ξ_gpsindy, hist_nvars = gpsindy( t, dx_stand_noise, Θx, 2*λ, α, ρ, abstol, reltol )  

Θx = pool_data_test( x_stand_GP, n_vars, poly_order ) 
Ξ_gpsindy_GP, hist_nvars = gpsindy( t, dx_stand_GP, Θx, 2*λ, α, ρ, abstol, reltol )  

println( "(stand) true - SINDy err    = ", norm( Ξ_stand_true - Ξ_stand_sindy )  ) 
# println( "(stand) true - SINDy_GP err = ", norm( Ξ_stand_true - Ξ_stand_sindy_GP )  ) 
println( "(stand) true - GPSINDy err  = ", norm( Ξ_stand_true - Ξ_gpsindy )  ) 
println( "(stand) true - GPSINDy_GP err  = ", norm( Ξ_stand_true - Ξ_gpsindy_GP )  ) 


## ============================================ ##
# NON-STANDARDIZED: try various SINDy's 

Ξ_true  = SINDy_test( x_true, dx_true, λ ) 
Ξ_sindy = SINDy_test( x_noise, dx_noise, λ ) 

# x_GP = post_dist_SE( t, x_noise, t )
# dx_GP  = fdiff( t, x_GP, 2 ) 

# Ξ_sindy_GP = SINDy_test( x_GP, dx_GP, λ ) 

# ----------------------- #
# using GPSINDy 

# # placeholder 
# Θx = pool_data_test( x_GP, n_vars, poly_order ) 
# Ξ_gpsindy_GP, hist_nvars = gpsindy( t, dx_GP, Θx, 3*λ, α, ρ, abstol, reltol )  

# placeholder 
Θx = pool_data_test( x_noise, n_vars, poly_order ) 
Ξ_gpsindy, hist_nvars = gpsindy( t, dx_noise, Θx, λ, α, ρ, abstol, reltol )  

println( "true - SINDy err    = ", norm( Ξ_true - Ξ_sindy )  ) 
# println( "true - SINDy_GP err = ", norm( Ξ_true - Ξ_sindy_GP )  ) 
# println( "true - GPSINDy err  = ", norm( Ξ_true - Ξ_gpsindy_GP )  ) 
println( "true - GPSINDy err  = ", norm( Ξ_true - Ξ_gpsindy )  ) 

dx_sindy   = Θx * Ξ_sindy 
dx_gpsindy = Θx * Ξ_gpsindy 

plot!( t, dx_sindy[:,2], ls = :dash, label = "SINDy", c = :red ) 


## ============================================ ##
# use GP to smooth data 


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
