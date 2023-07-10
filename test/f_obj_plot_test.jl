

#  
fn             = predator_prey 
plot_option    = 1 
savefig_option = 0 
fd_method      = 2 # 1 = forward, 2 = central, 3 = backward 

# choose ODE, plot states --> measurements 
x0, dt, t, x, dx_true, dx_fd = ode_states(fn, plot_option, fd_method) 


## ============================================ ##


    # ----------------------- #
    # SINDy 

    n_vars = size(x, 2) 
    poly_order = n_vars 

    # construct data library 
    Θx = pool_data_test(x, n_vars, poly_order) 

    # first cut - SINDy 
    Ξ = sparsify_dynamics_test(Θx, dx_fd, λ, n_vars) 

    # ----------------------- #
    # objective function 

    z_soln = 0 * Ξ 

    # ADMM stuff 
    ρ = 1.0 
    α = 1.0 

    # ----------------------- #
    # loop with state j

    j = 1 

    # initial loss function vars 
    ξ  = 0 * Ξ[:,j] 
    dx = dx_fd[:,j] 

    # assign for f_hp_opt 
    f_hp(ξ, σ_f, l, σ_n) = f_obj( σ_f, l, σ_n, dx, ξ, Θx )

## ============================================ ##
# plot effects 

ξ = Ξ_true[:,1]

# default 
σ_f = 1.0 
l   = 1.0 
σ_n = 0.1 

σ_vec = collect( 0 : 0.1 : 10 )

## ============================================ ## 

f(σ_f) = f_hp(ξ, σ_f, l, σ_n) 
p_σ_f = plot( σ_vec, f )
    plot!( title = "\n Varying σ_f " )

f(l) = f_hp(ξ, σ_f, l, σ_n) 
p_l = plot( σ_vec, f )
    plot!( title = "\n Varying l " ) 

f(σ_n) = f_hp(ξ, σ_f, l, σ_n) 
p_σ_n = plot( σ_vec, f )
    plot!( title = "\n Varying σ_n " )

p_hp = [ p_σ_f, p_l, p_σ_n ]
plot( p_hp ... , 
    layout = (1,3), 
    size   = [ 800 250 ], 
    margin = 5Plots.mm,
    bottom_margin = 7Plots.mm,  
    plot_title = string("Default [ σ_f, l, σ_n ] = [ " , σ_f, ", ", l, ", ", σ_n, " ] " ), 
    )

## ============================================ ##

f( σ_f, l ) = f_hp(ξ, σ_f, l, σ_n) 
p_σ_f_l = surface( σ_vec, σ_vec, f )
    plot!( xlabel = "σ_f", ylabel = "l", zlabel = "f", title = "\n Varying σ_f and l" )

f( l, σ_n ) = f_hp(ξ, σ_f, l, σ_n) 
p_l_σ_n = surface( σ_vec, σ_vec, f )
    plot!( xlabel = "l", ylabel = "σ_n", zlabel = "f", title = "\n Varying l and σ_n" )

f( σ_f, σ_n ) = f_hp(ξ, σ_f, l, σ_n) 
p_σ_f_σ_n = surface( σ_vec, σ_vec, f )
    plot!( xlabel = "σ_f", ylabel = "σ_n", zlabel = "f", title = "\n Varying σ_f and σ_n" )
        
p_hp = [ p_σ_f_l, p_l_σ_n, p_σ_f_σ_n ]
plot( p_hp ... , 
    layout = ( 1,3), 
    size   = [ 800 250 ], 
    margin = 5Plots.mm,
    bottom_margin = 7Plots.mm,  
    plot_title = string("Default [ σ_f, l, σ_n ] = [ " , σ_f, ", ", l, ", ", σ_n, " ] " ), 
    )
    