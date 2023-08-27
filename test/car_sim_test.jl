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
Ξ_gpsindy = SINDy_test( x_GP, dx_GP, λ, data_train.u ) 
Ξ_gpsindy_terms = pretty_coeffs(Ξ_GP, x_GP, data_train.u) 

## ============================================ ##
# validate 

using LinearAlgebra

z_fd = Ξ_true 

# get # states AND control inputs 
x_vars = size( data_train.dx_true, 2 ) 
u_vars = size( data_train.u, 2 ) 
n_vars = size( data_train.dx_true, 2 ) + size( data_train.u, 2 )  

# define pool_data functions 
fn_vector = pool_data_vecfn_test(n_vars, poly_order) 

# numerically evaluate each function at x and return a vector of numbers
𝚽( x, fn_vector ) = [ f(x) for f in fn_vector ]

# create vector of functions, each element --> each state 
dx_fn_vec = Vector{Function}(undef,0) 
for i = 1:x_vars 
    # define the differential equation 
    push!( dx_fn_vec, (x,p,t) -> dot( 𝚽( x, fn_vector ), z_fd[:,i] ) ) 
end 

dx_fn(x,p,t) = [ f(x,p,t) for f in dx_fn_vec ] 


## ============================================ ##


poly_order = 3 
dx_sindy_fn      = build_dx_fn(poly_order, n_vars, Ξ_sindy) 
dx_gpsindy_fn    = build_dx_fn(poly_order, n_vars, Ξ_gpsindy) 

dt = data_train.t[2] - data_train.t[1] 
t_sindy_val,   x_sindy_val   = validate_data( data_test.t, [ data_test.x_noise data_test.u ], dx_sindy_fn, dt ) 
t_gpsindy_val, x_gpsindy_val = validate_data( data_test.t, x_GP_train, dx_gpsindy_fn, dt ) 








