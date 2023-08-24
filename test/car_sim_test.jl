using GaussianSINDy 

## ============================================ ##
# truth 

fn = unicycle 

x0, dt, t, x_true, dx_true, dx_fd, p = ode_states(fn, 0, 2) 

# noise 
noise = 0.01 
x_noise  = x_true + noise*randn( size(x_true, 1), size(x_true, 2) )
dx_noise = dx_true + noise*randn( size(dx_true, 1), size(dx_true, 2) )

u = [] 
for i = 1 : length(t) 
    push!( u, [ 1/2*sin(t[i]), cos(t[i]) ] ) 
end 
u = vv2m(u) 

# split into training and test data 
test_fraction = 0.2 
portion       = 5 
t_train,        t_test        = split_train_test(t,        test_fraction, portion) 
x_train_true,   x_test_true   = split_train_test(x_true,   test_fraction, portion) 
dx_train_true,  dx_test_true  = split_train_test(dx_true,  test_fraction, portion) 
x_train_noise,  x_test_noise  = split_train_test(x_noise,  test_fraction, portion) 
dx_train_noise, dx_test_noise = split_train_test(dx_noise, test_fraction, portion) 

λ = 0.1 
Ξ_true = SINDy_test( x_train_true, dx_train_true, λ, u ) 
Ξ_true_terms = pretty_coeffs(Ξ_train_true, x_train_true, u) 

Ξ_noise = SINDy_test( x_noise, dx_noise, λ, u ) 
Ξ_noise_terms = pretty_coeffs(Ξ_noise, x_noise, u) 

# GPSINDy 
x_GP  = gp_post( t, 0*x_noise, t, 0*x_noise, x_noise ) 
dx_GP = gp_post( x_GP, 0*dx_noise, x_GP, 0*dx_noise, dx_noise ) 
Ξ_GP  = SINDy_test( x_GP, dx_GP, λ, u ) 
Ξ_GP_terms = pretty_coeffs(Ξ_GP, x_GP, u) 

