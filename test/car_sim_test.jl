using GaussianSINDy 

## ============================================ ##
# truth 

fn = unicycle 
data_train, data_test = ode_train_test( fn ) 

λ = 0.1 
Ξ_true = SINDy_test( data_train.x_true, data_train.dx_true, λ, data_train.u ) 
Ξ_true_terms = pretty_coeffs(Ξ_true, data_train.x_true, data_train.u) 

Ξ_sindy = SINDy_test( data_train.x_noise, data_train.dx_noise, λ, data_train.u ) 
Ξ_sindy_terms = pretty_coeffs(Ξ_sindy, data_train.x_noise, data_train.u) 

# GPSINDy 
x_GP  = gp_post( data_train.t, 0*data_train.x_noise, data_train.t, 0*data_train.x_noise, data_train.x_noise ) 
dx_GP = gp_post( x_GP, 0*data_train.dx_noise, x_GP, 0*data_train.dx_noise, data_train.dx_noise ) 
Ξ_gpsindy  = SINDy_test( x_GP, dx_GP, λ, data_train.u ) 
Ξ_gpsindy_terms = pretty_coeffs(Ξ_GP, x_GP, data_train.u) 







