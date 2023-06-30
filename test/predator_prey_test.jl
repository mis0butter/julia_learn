
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
test_fraction = 0.2 
portion       = 5 
t_train, t_test             = split_train_test(t, test_fraction, portion) 
x_train, x_test             = split_train_test(x, test_fraction, portion) 
dx_true_train, dx_true_test = split_train_test(dx_true, test_fraction, portion) 
dx_fd_train, dx_fd_test     = split_train_test(dx_fd, test_fraction, portion) 


## ============================================ ##
# SINDy alone 

λ = 0.1  
n_vars     = size(x, 2) 
poly_order = n_vars 

Ξ_sindy = SINDy_c_recursion(x, dx_fd, 0, λ, poly_order ) 
Ξ_true  = SINDy_c_recursion(x, dx_true, 0, λ, poly_order ) 


## ============================================ ##
# SINDy + GP + ADMM 

λ = 0.02875 
println("λ = ", λ)

# finite difference 
hist_fd = Hist( [], [], [], [], [] ) 
@time z_gpsindy, hist_fd = sindy_gp_admm( x_train, dx_fd_train, λ, hist_fd ) 
# display(z_gpsindy) 

# plot 
plot_admm(hist_fd, 1)
plot_admm(hist_fd, 2)

## ============================================ ## 
# generate + validate TEST data 

dx_gpsindy_fn = build_dx_fn(poly_order, z_gpsindy) 
dx_sindy_fn   = build_dx_fn(poly_order, Ξ_sindy)

t_gpsindy_val, x_gpsindy_val = validate_data(t_test, x_test, dx_gpsindy_fn, dt) 
t_sindy_val,   x_sindy_val   = validate_data(t_test, x_test, dx_sindy_fn, dt) 

# plot!! 
plot_prey_predator( t_train, x_train, t_test, x_test, t_sindy_val, x_sindy_val, t_gpsindy_val, x_gpsindy_val ) 

plot_test_data( t_test, x_test, t_sindy_val, x_sindy_val, t_gpsindy_val, x_gpsindy_val ) 

## ============================================ ##
# print stats 

if isequal( t_test[end], t_sindy_val[end] ) && isequal( t_test[end], t_gpsindy_val[end] )
    
    # print x0 
    println("x0 = ", round.(x0, digits = 2))

    function diff_norm( Ξ_true, Ξ_sindy, i )
        output = norm( Ξ_true[:,i] - Ξ_sindy[:,i] )
        return output 
    end 

    table_norm = []
    # ----------------------- #
    # true - SINDy 

    # coeff 1 
    push!(table_norm, diff_norm( Ξ_true, Ξ_sindy, 1 )) 
    # coeff 2 
    push!(table_norm, diff_norm( Ξ_true, Ξ_sindy, 2 ))
    # dyn 1 
    push!(table_norm, diff_norm( x_test, x_sindy_val, 1 ))
    # dyn 2 
    push!(table_norm, diff_norm( x_test, x_sindy_val, 2 ))

    # ----------------------- #
    # true - GPSINDy 

    # coeff 1 
    push!(table_norm, diff_norm( Ξ_true, z_gpsindy, 1 ))
    # coeff 2 
    push!(table_norm, diff_norm( Ξ_true, z_gpsindy, 2 ))
    # dyn 1 
    push!(table_norm, diff_norm( x_test, x_gpsindy_val, 1 ))
    # dyn 2 
    push!(table_norm, diff_norm( x_test, x_gpsindy_val, 2 ))

    # ----------------------- #
    # for displaying in table 

    table_norm = round.(table_norm, digits = 2)
    display(table_norm[:,:]') 

else

    println("poop something diverged" )

end 

## ============================================ ##
# save fig 

using Plots 

if savefig_option == 1
    savefig("./test/outputs/sindy_gpsindy.pdf")
end 

# jldsave("test/outputs/test.jld2"; t, x, z_gpsindy, hist_fd)

