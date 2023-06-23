
struct Hist 
    objval 
    r_norm 
    s_norm 
    eps_pri 
    eps_dual 
end 

using GaussianSINDy
using LinearAlgebra 

## ============================================ ##
# choose ODE, plot states --> measurements 

#  
fn             = predator_prey 
plot_option    = 1 
savefig_option = 0 

# choose ODE, plot states --> measurements 
x0, dt, t, x, dx_true, dx_fd = ode_states(fn, plot_option) 

# split into training and validation data 
train_fraction = 0.7 
t_train, t_test             = split_train_test(t, train_fraction) 
x_train, x_test             = split_train_test(x, train_fraction) 
dx_true_train, dx_true_test = split_train_test(dx_true, train_fraction) 
dx_fd_train, dx_fd_test     = split_train_test(dx_fd, train_fraction) 


## ============================================ ##
# SINDy alone 

λ = 0.1  
n_vars     = size(x, 2) 
poly_order = n_vars 

Ξ_sindy = SINDy_c_recursion(x, dx_fd, 0, λ, poly_order ) 
Ξ_true  = SINDy_c_recursion(x, dx_true, 0, λ, poly_order ) 


## ============================================ ##
# SINDy + GP + ADMM 

λ = 0.02  
println("λ = ", λ)

# finite difference 
hist_fd = Hist( [], [], [], [], [] ) 
@time z_gpsindy, hist_fd = sindy_gp_admm( x_train, dx_fd_train, λ, hist_fd ) 
display(z_gpsindy) 


## ============================================ ## 
# generate + validate data 

dx_gpsindy_fn = build_dx_fn(poly_order, z_gpsindy) 
dx_sindy_fn   = build_dx_fn(poly_order, Ξ_sindy)

t_gpsindy_val, x_gpsindy_val = validate_data(t_test, x_test, dx_gpsindy_fn, dt) 
t_sindy_val, x_sindy_val     = validate_data(t_test, x_test, dx_sindy_fn, dt) 

# plot!! 
plot_prey_predator( t_train, x_train, t_test, x_test, t_sindy_val, x_sindy_val, t_gpsindy_val, x_gpsindy_val ) 

## ============================================ ##
# print stats 

# print some coeff stats 
coeff_norm = [ opnorm( Ξ_true - Ξ_sindy ) , opnorm( Ξ_true - z_gpsindy ) ] 
println("COEFFICIENTS")
println("  opnorm( Ξ_true - Ξ_sindy ) = \n    ", coeff_norm[1] )
println("  opnorm( Ξ_true - z_gpsindy ) = \n    ", coeff_norm[2] )

# print some predicted stats 
valid_norm = zeros(1, 4) 
println("VALIDATION DATA")

    println("  i = ", i) 
    i = 1 
    # true - SINDy 
    valid_norm[1,1] = norm( x_test[:,i] - x_sindy_val[:,i] ) 
    println("    opnorm( x_true - x_sindy ) = \n    ", valid_norm[1,1] )

    # true - GP SINDy
    valid_norm[1,2] = norm( x_test[:,i] - x_gpsindy_val[:,i] )
    println("    opnorm( x_true - x_gpsindy ) = \n    ", valid_norm[1,2] )     

    println("  i = ", 2) 
    i = 2 
    # true - SINDy 
    valid_norm[1,3] = norm( x_test[:,i] - x_sindy_val[:,i] ) 
    println("    opnorm( x_true - x_sindy ) = \n    ", valid_norm[1,3] )

    # true - GP SINDy
    valid_norm[1,4] = norm( x_test[:,i] - x_gpsindy_val[:,i] )
    println("    opnorm( x_true - x_gpsindy ) = \n    ", valid_norm[1,4] )     

    println("x0 = ", round.(x0, digits = 2))

    # for displaying in table 
    table_norm = round.([ coeff_norm[1] valid_norm[1] valid_norm[2] coeff_norm[2] valid_norm[3]  valid_norm[4] ], digits = 2)


## ============================================ ##
# save fig 

if savefig_option == 1
    savefig("./sindy_gpsindy.pdf")
end 



