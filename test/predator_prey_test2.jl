
struct Hist 
    objval 
    fval 
    gval 
    hp 
    r_norm 
    s_norm 
    eps_pri 
    eps_dual 
end 

using GaussianSINDy
using LinearAlgebra 
using Plots 

## ============================================ ##
# choose ODE, plot states --> measurements 

#  
fn             = predator_prey 
plot_option    = 1 
savefig_option = 0 
fd_method      = 2 # 1 = forward, 2 = central, 3 = backward 

# choose ODE, plot states --> measurements 
x0, dt, t, x, dx_true, dx_fd = ode_states(fn, plot_option, fd_method) 

# truth coeffs 
Ξ_true = SINDy_test( x, dx_true, 0.1 ) 
Ξ_true = Ξ_true[:,1] 

dx_noise_vec = 0 : 0.1 : 1.0  
dx_noise = 0.0

Ξ_sindy_vec         = [] 
Ξ_sindy_err_vec     = [] 
z_gpsindy_vec       = [] 
z_gpsindy_err_vec   = [] 
hist_gpsindy_vec    = [] 
for dx_noise = dx_noise_vec 
    println("dx_noise = ", dx_noise)

    Ξ_sindy, Ξ_sindy_err, z_gpsindy, z_gpsindy_err, hist_gpsindy = monte_carlo_gpsindy( x0, dt, t, x, dx_true, dx_fd, dx_noise )

    println( "  Ξ_sindy = " );          println( Ξ_sindy )
    println( "  z_gpsindy = " );        println( z_gpsindy )
    println( "  Ξ_sindy_err = " );      println( Ξ_sindy_err )
    println( "  z_gpsindy_err = " );    println( z_gpsindy_err )

    push!(Ξ_sindy_vec, Ξ_sindy) 
    push!(z_gpsindy_vec, z_gpsindy)
    push!(Ξ_sindy_err_vec, Ξ_sindy_err) 
    push!(z_gpsindy_err_vec, z_gpsindy_err)
    push!( hist_gpsindy_vec, hist_gpsindy )
end 

Ξ_sindy_err_vec   = mapreduce(permutedims, vcat, Ξ_sindy_err_vec)
z_gpsindy_err_vec = mapreduce(permutedims, vcat, z_gpsindy_err_vec)


## ============================================ ##
# ----------------------- #
using Plots 

p_Ξ = [] 
for i = 1:2
    p_ξ = plot( dx_noise_vec, Ξ_sindy_err_vec[:,i], label = "SINDy" )
    plot!( p_ξ, dx_noise_vec, z_gpsindy_err_vec[:,i], ls = :dash, label = "GPSINDy" )
    plot!( p_ξ, 
        legend = false, 
        xlabel = "dx_noise", 
        title  = string( "ξ", i, "_true - ξ", i, "_discovered" ), 
        ) 
    push!(p_Ξ, p_ξ)
end 
p = deepcopy(p_Ξ[end])  
plot!(p, 
    legend     = (-0.2,0.6) , 
    framestyle = :none , 
    title      = "", 
    )
push!( p_Ξ, p ) 
p_Ξ = plot(p_Ξ ... , 
    layout = grid( 1, 3, widths=[0.45, 0.45, 0.45] ) , 
    size   = [ 800 300 ], 
    ) 
display(p_Ξ)

t_str = string( "dx_true + dx_noise*randn \n dx_noise = ", minimum( dx_noise_vec ), " --> ", maximum( dx_noise_vec ) )
p_noise = plot( t, dx_true[:,1], title = t_str, xlabel = "Time (s)" )
for dx_noise = dx_noise_vec
    plot!(p_noise, t, dx_true[:,1] .+ dx_noise*randn( size(dx_true, 1), 1 ) ) 
end 
display(p_noise)


## ============================================ ##
# save data  

using JLD2

savefig(p_Ξ, "./test/outputs/p_Ξ_3nonlin.pdf")
savefig(p_noise, "./test/outputs/p_noise_3nonlin.pdf")

jldsave("test/outputs/batch_results_3nonlin.jld2"; t, dx_true, Ξ_sindy_err_vec, z_gpsindy_err_vec, dx_noise_vec )


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
    hist_fd = Hist( [], [], [], [], [], [], [], [] ) 
    @time z_gpsindy, hist_fd = sindy_gp_admm( x_train, dx_fd_train, λ, hist_fd ) 
    # display(z_gpsindy) 

    Ξ_sindy_err   = [ norm( Ξ_true[:,1] - Ξ_sindy[:,1] ), norm( Ξ_true[:,2] - Ξ_sindy[:,2] )  ] 
    z_gpsindy_err = [ norm( Ξ_true[:,1] - z_gpsindy[:,1] ), norm( Ξ_true[:,2] - z_gpsindy[:,2] )  ] 

    # return Ξ_sindy, z_gpsindy
    return Ξ_sindy, Ξ_sindy_err, z_gpsindy, z_gpsindy_err, hist_fd 

end 

