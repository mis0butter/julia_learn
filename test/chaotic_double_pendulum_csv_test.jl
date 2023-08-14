using CSV 
using DataFrames 

## ============================================ ##

csv_file = "test/data/double-pendulum-chaotic/original/dpc_dataset_csv/0.csv" 

# wrap in data frame --> Matrix 
df = CSV.read(csv_file, DataFrame) 
data = Matrix(df) 


## ============================================ ##

x_red   = data[:,1] 
y_red   = data[:,2] 
x_green = data[:,3] 
y_green = data[:,4] 
x_blue  = data[:,5] 
y_blue  = data[:,6]

x1 = x_green - x_red 
y1 = y_green - y_red 
x2 = x_blue - x_green 
y2 = y_blue - y_green 

r1 = sqrt.( x1.^2 + y1.^2 ) 
r2 = sqrt.( x2.^2 + y2.^2 )
theta1 = asin.( x1./r1 )  
theta2 = asin.( x2./r2 )  
theta  = [theta1 theta2]

dt = 1/60       # best guess 
t_train  = 0 : dt : dt * size(data, 1) - dt
t_train  = collect(t_train) 
dtheta = fdiff( t_train, theta, 2 ) 

x_train  = [ theta[:,1] dtheta[:,1] theta[:,2] dtheta[:,2] ] 
dx_train = fdiff(t_train, x_train, 2) 



## ============================================ ##


# SINDy by itself 
Θx_sindy = pool_data_test( x_train, n_vars, poly_order ) 
Ξ_sindy  = SINDy_test( x_train, dx_train, λ ) 
 

# ----------------------- #
# GPSINDy (first) 

# step -1 : smooth x measurements with t (temporal)  
x_train_GP, Σ_xsmooth, hp   = post_dist_SE( t_train, x_train, t_train )  

# step 0 : smooth dx measurements with x_GP (non-temporal) 
dx_train_GP, Σ_dxsmooth, hp = post_dist_SE( x_train_GP, dx_train, x_train_GP )  
# dx_test = gp_post( x_train_GP, 0*dx_train, x_train_GP, 0*dx_train, dx_train ) 

# SINDy 
Θx_gpsindy = pool_data_test(x_train_GP, n_vars, poly_order) 
Ξ_gpsindy  = SINDy_test( x_train_GP, dx_train_GP, λ ) 

# ----------------------- #
# GPSINDy (second) 

# step 2: GP 
dx_mean = Θx_gpsindy * Ξ_gpsindy 
dx_post = gp_post( x_train_GP, dx_mean, x_train_GP, dx_mean, dx_train ) 

# step 3: SINDy 
Θx_gpsindy   = pool_data_test( x_train_GP, n_vars, poly_order ) 
Ξ_gpsindy_x2 = SINDy_test( x_train_GP, dx_post, λ ) 

