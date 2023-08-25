using GaussianSINDy
using CSV 
using DataFrames 

# ----------------------- #
# load data 

csv_file = "test/data/jake_robot_data.csv" 

# wrap in data frame --> Matrix 
df   = CSV.read(csv_file, DataFrame) 
data = Matrix(df) 

# extract variables 
t = data[:,1] 
x = data[:,2:end-2]
u = data[:,end-1:end]

# use forward finite differencing 
dx_fd = fdiff(t, x, 1) 
# dx_true = dx_true_fn

# massage data, generate rollovers  
rollover_up_ind = findall( x -> x > 100, dx_fd[:,4] ) 
rollover_dn_ind = findall( x -> x < -100, dx_fd[:,4] ) 
for i = 1 : length(rollover_up_ind) 

    i0   = rollover_up_ind[i] + 1 
    ifin = rollover_dn_ind[i] 
    rollover_rng = x[ i0 : ifin , 4 ]
    dθ = π .- rollover_rng 
    θ  = -π .- dθ 
    x[ i0 : ifin , 4 ] = θ

end 

## ============================================ ##

# split into training and test data 
test_fraction = 0.2 
portion       = 5 
u_train,  u_test  = split_train_test( u, test_fraction, portion ) 
t_train,  t_test  = split_train_test( t, test_fraction, portion ) 
x_train,  x_test  = split_train_test( x, test_fraction, portion ) 
dx_train, dx_test = split_train_test( dx_fd, test_fraction, portion ) 
u_train,  u_test  = split_train_test( dx_fd, test_fraction, portion ) 



## ============================================ ##


# x_GP,  Σ_xGP,  hp = post_dist_SE( t_train, x_train, t_train )              # step -1 
# dx_GP, Σ_dxGP, hp = post_dist_SE( x_GP, dx_train, x_GP )    # step 0 

x_GP_train  = gp_post( t_train, 0*x_train, t_train, 0*x_train, x_train ) 
dx_GP_train = gp_post( x_GP_train, 0*dx_train, x_GP_train, 0*dx_train, dx_train ) 

x_GP_test   = gp_post( t_test, 0*x_test, t_test, 0*x_test, x_test ) 
dx_GP_test  = gp_post( x_GP_test, 0*dx_test, x_GP_test, 0*dx_test, dx_test ) 

## ============================================ ##



## ============================================ ##
# SINDy vs. GPSINDy 

n_vars = size( [x_train u_train], 2 )
x_vars = size(x_train, 2)
u_vars = size(u_train, 2) 
poly_order = n_vars 

λ = 0.1 
# Ξ = SINDy_c_test( x, u, dx_fd, λ ) 
Ξ_sindy       = SINDy_test( x_train, dx_train, λ, u_train ) 
Ξ_sindy_terms = pretty_coeffs( Ξ_sindy, x_train, u_train ) 

Ξ_gpsindy       = SINDy_test( x_GP_train, dx_GP_train, λ, u_train ) 
Ξ_gpsindy_terms = pretty_coeffs( Ξ_gpsindy, x_GP_train, u_train ) 


## ============================================ ##

using Plots 

# SINDy alone 
Θx = pool_data_test( [x_train u_train], n_vars, poly_order) 
# Ξ_sindy = sparsify_dynamics_test( Θx, dx_fd, λ, x_vars ) 
dx_sindy = Θx * Ξ_sindy 

# GPSINDy 
Θx = pool_data_test( [x_GP_train u_train], n_vars, poly_order) 
# Ξ_gpsindy = sparsify_dynamics_test( Θx, dx_GP, λ, x_vars ) 
dx_gpsindy = Θx * Ξ_gpsindy 

plt = plot( title = "dx: meas vs. sindy", legend = :outerright )
scatter!( plt, t_train, dx_train[:,1], c = :black, ms = 3, label = "meas (finite diff)" )
plot!( plt, t_train, dx_GP_train[:,1], c = :blue, label = "GP" )
plot!( plt, t_train, dx_sindy[:,1], c = :red, ls = :dash, label = "SINDy" )   
plot!( plt, t_train, dx_gpsindy[:,1], c = :green, ls = :dashdot, label = "GPSINDy" )   

## ============================================ ##

dx_sindy_fn      = build_dx_fn(poly_order, Ξ_sindy) 
dx_gpsindy_fn    = build_dx_fn(poly_order, Ξ_gpsindy) 

t_sindy_val,      x_sindy_val      = validate_data(t_test, x_test, dx_sindy_fn, dt) 
# t_sindy_val,      x_sindy_val      = validate_data(t_test, x_test, fn, dt) 
t_gpsindy_val,    x_gpsindy_val    = validate_data(t, x_GP_train, dx_gpsindy_fn, dt) 

# plot!! 
plot_states( t, x, t, x, t_sindy_val, x_sindy_val, t_gpsindy_val, x_gpsindy_val, t_gpsindy_val, x_gpsindy_val ) 


