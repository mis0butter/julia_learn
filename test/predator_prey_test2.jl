
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
fd_method      = 2 # 1 = forward, 2 = central, 3 = backward 

# choose ODE, plot states --> measurements 
x0, dt, t, x, dx_true, dx_fd = ode_states(fn, plot_option, fd_method) 

Ξ_true = SINDy_test( x, dx_true, 0.1 ) 
Ξ_true = Ξ_true[:,1] 

dx_noise_vec = 0 : 0.02 : 0.5 
Ξ_sindy_err_vec   = [] 
z_gpsindy_err_vec = [] 
for dx_noise = dx_noise_vec 
    println("dx_noise = ", dx_noise)

    Ξ_sindy, Ξ_sindy_err, z_gpsindy, z_gpsindy_err = monte_carlo_gpsindy( x0, dt, t, x, dx_true, dx_fd, dx_noise )

    println( "  Ξ_sindy = " );   display( Ξ_sindy )
    println( "  z_gpsindy = " ); display( z_gpsindy )
    println( "  Ξ_sindy_err = " );   display( Ξ_sindy_err )
    println( "  z_gpsindy_err = " ); display( z_gpsindy_err )
    push!(Ξ_sindy_err_vec, Ξ_sindy_err) 
    push!(z_gpsindy_err_vec, z_gpsindy_err)
end 

p_Ξ = plot( dx_noise_vec, Ξ_sindy_err_vec, label = "SINDy" )
    plot!( dx_noise_vec, z_gpsindy_err_vec, ls = :dash, label = "GPSINDy" )
    plot!( 
        legend = true, 
        xlabel = "dx_noise", 
        title  = "Ξ_true - Ξ_discovered" 
        ) 
display(p) 

t_str = string( "dx_true + dx_noise*randn \n dx_noise = ", minimum( dx_noise_vec ), " --> ", maximum( dx_noise_vec ) )
p_noise = plot( t, dx_true[:,1], title = t_str, xlabel = "Time (s)" )
for dx_noise = dx_noise_vec
    plot!( t, dx_true[:,1] .+ dx_noise*randn( size(dx_true, 1), 1 ) ) 
end 
display(p_noise)


## ============================================ ## 

function monte_carlo_gpsindy(x0, dt, t, x, dx_true, dx_fd, dx_noise) 

    # HACK - adding noise to truth derivatives 
    dx_fd = dx_true .+ dx_noise*randn( size(dx_true, 1), size(dx_true, 2) ) 
    # dx_fd = dx_true 

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

    Ξ_true  = SINDy_test( x, dx_true, λ ) 
    Ξ_sindy = SINDy_test( x, dx_fd, λ ) 


    ## ============================================ ##
    # SINDy + GP + ADMM 

    λ = 0.02 

    # finite difference 
    hist_fd = Hist( [], [], [], [], [] ) 
    @time z_gpsindy, hist_fd = sindy_gp_admm( x_train, dx_fd_train, λ, hist_fd ) 
    # display(z_gpsindy) 

    dx1_sindy_err = norm( Ξ_true[:,1] - Ξ_sindy[:,1] ) 
    dx1_gpsindy_err = norm( Ξ_true[:,1] - z_gpsindy[:,1] ) 

    # return Ξ_sindy, z_gpsindy
    return Ξ_sindy[:,1], dx1_sindy_err, z_gpsindy[:,1], dx1_gpsindy_err  

end 

