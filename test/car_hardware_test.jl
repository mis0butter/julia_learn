using GaussianSINDy
using CSV 
using DataFrames 
using PrettyTables 


## ============================================ ##
# load data 


csv_file = "test/jake_robot_data.csv" 

# wrap in data frame --> Matrix 
df = CSV.read(csv_file, DataFrame) 
data = Matrix(df) 

# extract variables 
t = data[:,1] 
x = data[:,2:end-2]
u = data[:,end-1:end]

dx_fd = fdiff(t, x, 2) 
# dx_true = dx_true_fn

## ============================================ ##
# SINDy alone 

λ = 0.01 

n_vars = size( [x u], 2 )
x_vars = size(x, 2)
u_vars = size(u, 2) 
poly_order = n_vars 

# construct data library 
Θx = pool_data_test( [x u], n_vars, poly_order) 

# first cut - SINDy 
Ξ = sparsify_dynamics_test( Θx, dx_fd, λ, x_vars ) 

x_sindy = Θx * Ξ 

plt = plot( title = "meas vs. sindy", legend = :outerright )
plot!( plt, t, x[:,1], c = :blue, label = "meas" )
plot!( plt, t, x_sindy[:,1], c = :red, label = "sindy" )   

