using GaussianSINDy 

## ============================================ ##
# truth 

fn = unicycle 
data_train, data_test = ode_train_test( fn ) 

λ = 0.1 
Ξ_true = SINDy_test( data_train, dx_train_true, λ, u ) 
Ξ_true_terms = pretty_coeffs(Ξ_train_true, x_train_true, u) 

Ξ_noise = SINDy_test( x_noise, dx_noise, λ, u ) 
Ξ_noise_terms = pretty_coeffs(Ξ_noise, x_noise, u) 

# GPSINDy 
x_GP  = gp_post( t, 0*x_noise, t, 0*x_noise, x_noise ) 
dx_GP = gp_post( x_GP, 0*dx_noise, x_GP, 0*dx_noise, dx_noise ) 
Ξ_GP  = SINDy_test( x_GP, dx_GP, λ, u ) 
Ξ_GP_terms = pretty_coeffs(Ξ_GP, x_GP, u) 

