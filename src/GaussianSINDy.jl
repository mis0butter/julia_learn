module GaussianSINDy

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

include("SINDy.jl")
include("GP_tools.jl")
include("lasso_admm.jl")
include("ode_fns.jl")  
include("utils.jl")
include("init_params.jl")
include("SINDy_test.jl")


## ============================================ ## 
# SINDy + GP objective function 

using LinearAlgebra

export f_obj 
function f_obj( σ_f, l, σ_n, dx, ξ, Θx )

    # training kernel function 
    Ky  = k_SE(σ_f, l, dx, dx) + σ_n^2 * I 
    # Ky  = k_SE(σ_f, l, dx, dx) + (0.1 + σ_n^2) * I 
    # Ky  = k_periodic(σ_f, l, 1.0, dx, dx) + (0.1 + σ_n^2) * I 

    while det(Ky) == 0 
        println( "det(Ky) = 0" )
        Ky += σ_n * I 
    end 

    term  = 1/2*( dx - Θx*ξ )'*inv( Ky )*( dx - Θx*ξ ) 
    
    # # ----------------------- #
    # # LU factorization 

    # # let's say x = inv(Ky)*( dx - Θx*ξ ), or x = inv(A)*b 
    # A = Ky 
    # b = ( dx - Θx*ξ ) 
    # C = cholesky(A) ; L = C.L ; U = C.U 

    # y = L \ b 
    # x = U \ y 
    
    # term  = 1/2*( dx - Θx*ξ )'*x

    # ----------------------- #
    term += 1/2*log( tr(Ky) ) 

    return term 

end 


## ============================================ ##

export sindy_gp_admm 
function sindy_gp_admm( t, x, dx_fd, λ, hist_hp_opt )

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

    for j = 1 : n_vars 

        # initial loss function vars 
        ξ  = 0 * Ξ[:,j] 
        dx = dx_fd[:,j] 

        # assign for f_hp_opt 
        f_hp(ξ, σ_f, l, σ_n) = f_obj( σ_f, l, σ_n, dx, ξ, Θx )

        # l1 norm 
        g(z) = λ * sum(abs.(z)) 

        # ----------------------- #
        # admm!!! 

        n = length(ξ)
        println("t size = ", size(t))
        println( "Θx size = ", size(Θx) )
        # x_hp_opt, z_hp_opt, hist_hp_opt, k  = lasso_admm_hp_opt( t, f_hp, g, n, λ, ρ, α, hist_hp_opt ) 
        x_hp_opt, z_hp_opt, hist_hp_opt, k  = lasso_admm_gp_opt( t, dx, Θx, f_hp, g, n, λ, ρ, α, hist_hp_opt ) 

        # ----------------------- #
        # output solution 

        z_soln[:,j] = z_hp_opt 

    end 

    return z_soln, hist_hp_opt 

end 


## ============================================ ##

export monte_carlo_gpsindy 
function monte_carlo_gpsindy(x0, dt, t, x, dx_true, dx_fd, dx_noise, λ_gpsindy) 

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

    # λ = 0.02 

    # finite difference 
    hist_fd = Hist( [], [], [], [], [], [], [], [] ) 

    println( "t_train size = ", size(t_train) )
    println( "dx_fd_train size = ", size(dx_fd_train) )
    @time z_gpsindy, hist_fd = sindy_gp_admm( t_train, x_train, dx_fd_train, λ_gpsindy, hist_fd ) 
    # display(z_gpsindy) 

    Ξ_sindy_err   = [ norm( Ξ_true[:,1] - Ξ_sindy[:,1] ), norm( Ξ_true[:,2] - Ξ_sindy[:,2] )  ] 
    z_gpsindy_err = [ norm( Ξ_true[:,1] - z_gpsindy[:,1] ), norm( Ξ_true[:,2] - z_gpsindy[:,2] )  ] 

    # return Ξ_sindy, z_gpsindy
    return Ξ_sindy, Ξ_sindy_err, z_gpsindy, z_gpsindy_err, hist_fd 

end 

## ============================================ ##

end 