using GaussianSINDy
using CSV 
using DataFrames 


# ----------------------- #
# load data 

csv_file = "test/data/jake_robot_data.csv" 

# wrap in data frame --> Matrix 
df = CSV.read(csv_file, DataFrame) 
data = Matrix(df) 

# extract variables 
t = data[:,1] 
x = data[:,2:end-2]
u = data[:,end-1:end]

dx_fd = fdiff(t, x, 2) 
# dx_true = dx_true_fn

# split into training and test data 
test_fraction = 0.2 
portion       = 5 
t_train, t_test   = split_train_test(t, test_fraction, portion) 
x_train,  x_test  = split_train_test(x, test_fraction, portion) 
dx_train, dx_test = split_train_test(dx_fd, test_fraction, portion) 
u_train, u_test   = split_train_test(dx_fd, test_fraction, portion) 

x_GP,  Σ_xGP,  hp = post_dist_SE( t_train, x_train, t_train )              # step -1 
dx_GP, Σ_dxGP, hp = post_dist_SE( x_GP, dx_train, x_GP )    # step 0 


## ============================================ ##
# SINDy vs. GPSINDy 

n_vars = size( [x_train u_train], 2 )
x_vars = size(x_train, 2)
u_vars = size(u_train, 2) 
poly_order = n_vars 

λ = 0.1 
# Ξ = SINDy_c_test( x, u, dx_fd, λ ) 
Ξ_sindy = SINDy_test( x_train, dx_train, λ, u ) 
Ξ_gpsindy = SINDy_test( x_GP, dx_GP, λ, u ) 


## ============================================ ##

using Plots 

# SINDy alone 
Θx = pool_data_test( [x u], n_vars, poly_order) 
# Ξ_sindy = sparsify_dynamics_test( Θx, dx_fd, λ, x_vars ) 
dx_sindy = Θx * Ξ_sindy 

# GPSINDy 
Θx = pool_data_test( [x_GP u], n_vars, poly_order) 
# Ξ_gpsindy = sparsify_dynamics_test( Θx, dx_GP, λ, x_vars ) 
dx_gpsindy = Θx * Ξ_gpsindy 

plt = plot( title = "dx: meas vs. sindy", legend = :outerright )
scatter!( plt, t, dx_fd[:,1], c = :black, ms = 3, label = "meas (finite diff)" )
plot!( plt, t, dx_GP[:,1], c = :blue, label = "GP" )
plot!( plt, t, dx_sindy[:,1], c = :red, ls = :dash, label = "SINDy" )   
plot!( plt, t, dx_gpsindy[:,1], c = :green, ls = :dashdot, label = "GPSINDy" )   

## ============================================ ##

dx_sindy_fn      = build_dx_fn(poly_order, Ξ_sindy) 
dx_gpsindy_fn    = build_dx_fn(poly_order, Ξ_gpsindy) 

t_sindy_val,      x_sindy_val      = validate_data(t, x, dx_sindy_fn, dt) 
# t_sindy_val,      x_sindy_val      = validate_data(t_test, x_test, fn, dt) 
t_gpsindy_val,    x_gpsindy_val    = validate_data(t, x_GP, dx_gpsindy_fn, dt) 

# plot!! 
plot_states( t, x, t, x, t_sindy_val, x_sindy_val, t_gpsindy_val, x_gpsindy_val, t_gpsindy_val, x_gpsindy_val ) 


