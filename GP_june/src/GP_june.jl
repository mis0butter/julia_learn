module GP_june 

include("SINDy.jl")
include("GP_tools.jl")
include("lasso_admm.jl")


## ============================================ ##
# SINDy + GP objective function 

export f_obj 
function f_obj(( σ_f, l, σ_n, dx, ξ, Θx ))

    # training kernel function 
    Ky  = k_fn((σ_f, l, dx, dx)) + σ_n^2 * I 

    term  = 1/2*( dx - Θx*ξ )'*inv( Ky )*( dx - Θx*ξ ) 
    term += 1/2*log(det( Ky )) 

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
        f_hp(ξ, σ_f, l, σ_n) = f_obj(( σ_f, l, σ_n, dx, ξ, Θx ))

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