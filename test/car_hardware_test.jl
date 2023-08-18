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

# x_GP,  Σ_xGP,  hp = post_dist_SE( t, x, t )              # step -1 
# dx_GP, Σ_dxGP, hp = post_dist_SE( x_GP, dx_fd, x_GP )    # step 0 


## ============================================ ##
# SINDy vs. GPSINDy 

n_vars = size( [x u], 2 )
x_vars = size(x, 2)
u_vars = size(u, 2) 
poly_order = n_vars 

λ = 0.1 
Ξ = SINDy_c_test( x, u, dx_fd, λ ) 


## ============================================ ##

# SINDy alone 
Θx = pool_data_test( [x u], n_vars, poly_order) 
Ξ_sindy = sparsify_dynamics_test( Θx, dx_fd, λ, x_vars ) 
dx_sindy = Θx * Ξ_sindy 

# GPSINDy 
Θx = pool_data_test( [x_GP u], n_vars, poly_order) 
Ξ_gpsindy = sparsify_dynamics_test( Θx, dx_GP, λ, x_vars ) 
dx_gpsindy = Θx * Ξ_gpsindy 

plt = plot( title = "dx: meas vs. sindy", legend = :outerright )
scatter!( plt, t, dx_fd[:,1], c = :black, ms = 3, label = "meas (finite diff)" )
plot!( plt, t, dx_GP[:,1], c = :blue, label = "GP" )
plot!( plt, t, dx_sindy[:,1], c = :red, ls = :dash, label = "SINDy" )   
plot!( plt, t, dx_gpsindy[:,1], c = :green, ls = :dashdot, label = "GPSINDy" )   

