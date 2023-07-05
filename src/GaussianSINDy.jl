module GaussianSINDy

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
    # Ky  = k_SE((σ_f, l, dx, dx)) + σ_n^2 * I 
    # Ky  = k_SE(σ_f, l, dx, dx) + (0.1 + σ_n^2) * I 
    Ky  = k_periodic(σ_f, l, 1.0, dx, dx) + (0.1 + σ_n^2) * I 

    term  = 1/2*( dx - Θx*ξ )'*inv( Ky )*( dx - Θx*ξ ) 
    
    # # if Ky really small 
    # if det(Ky) == 0 
    #     # e     = eigvals_june(Ky) 
    #     # e     = eigen(Ky).values 
    #     # log_e = log.(e) 
    #     # Ky    = sum(log_e) 
    #     # term += 1/2*log( tr(Ky) ) 
    #     println("det(Ky) = 0")
    #     term += 1/2*log( det(Ky) ) 
    # else
    #     term += 1/2*log( det(Ky) ) 
    # end 

    term += 1/2*log( tr(Ky) ) 

    return term 

end 


## ============================================ ##

export sindy_gp_admm 
function sindy_gp_admm( x, dx_fd, λ, hist_hp_opt )

    # ----------------------- #
    # SINDy 

    n_vars = size(x, 2) 
    poly_order = n_vars 

    # construct data library 
    Θx = pool_data(x, n_vars, poly_order) 

    # first cut - SINDy 
    Ξ = sparsify_dynamics(Θx, dx_fd, λ, n_vars) 

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
        x_hp_opt, z_hp_opt, hist_hp_opt, k  = lasso_admm_hp_opt( f_hp, g, n, λ, ρ, α, hist_hp_opt ) 

        # ----------------------- #
        # output solution 

        z_soln[:,j] = z_hp_opt 

    end 

    return z_soln, hist_hp_opt 

end 

end 

