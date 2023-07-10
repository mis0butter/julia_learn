
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
using Dates 
using Optim 
using GaussianProcesses

## ============================================ ##
# choose ODE, plot states --> measurements 

#  
fn             = predator_prey 
plot_option    = 0 
savefig_option = 0 
fd_method      = 2 # 1 = forward, 2 = central, 3 = backward 

# choose ODE, plot states --> measurements 
x0, dt, t, x, dx_true, dx_fd = ode_states(fn, plot_option, fd_method) 

# truth coeffs 
Ξ_true = SINDy_test( x, dx_true, 0.1 ) 
Ξ_true = Ξ_true[:,1] 
 
dx_noise  = 1.0  
λ = 0.02 

# ----------------------- #

    println("dx_noise = ", dx_noise)
    println("λ_gpsindy = ", λ)

# ----------------------- #
# MONTE CARLO GPSINDY 

    dx_fd = dx_true .+ dx_noise*randn( size(dx_true, 1), size(dx_true, 2) ) 

# ----------------------- #
# SINDy alone 

    λ = 0.1  
    n_vars     = size(x, 2) 
    poly_order = n_vars 

    Ξ_true  = SINDy_test( x, dx_true, λ ) 
    Ξ_sindy = SINDy_test( x, dx_fd, λ ) 

    λ = 0.02 

# ----------------------- #
# SINDy + GP + ADMM 

    # λ = 0.02 

    # finite difference 
    hist = Hist( [], [], [], [], [], [], [], [] ) 

    
## ============================================ ##
# sindy_gp_admm 

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
    # for j = 1 : n_vars 

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
        
## ============================================ ##
# LASSO ADMM GP OPT 


    # define constants 
    max_iter = 1000  
    abstol   = 1e-4 
    reltol   = 1e-2           # save matrix-vector multiply 

    # ADMM solver 
    ξ = z = u = zeros(n) 

    # initial hyperparameters 
    σ_f0 = log(1.0) ; σ_f = σ_f0  
    l_0  = log(1.0) ; l   = l_0   
    σ_n0 = log(0.1) ; σ_n = σ_n0 

    # augmented Lagrangian (scaled form) 
    aug_L(ξ, σ_f, l, σ_n, z, u) = f_hp(ξ, σ_f, l, σ_n) + g(z) + ρ/2 .* norm( ξ - z + u )^2 

    # counter 
    iter = 0 

    # update λ
    λ = log(f_hp( ξ, σ_f, l, σ_n )) 
    
    # begin iterations 
    k = 1 
    # for k = 1 : max_iter 

        # increment counter 
        iter += 1 


## ============================================ ##
        # x-update (optimization) 

        # optimization 
        f_opt(ξ) = aug_L(ξ, σ_f, l, σ_n, z, u) 
        od       = OnceDifferentiable( f_opt, ξ ; autodiff = :forward ) 
        result   = optimize( od, ξ, LBFGS() ) 
        ξ        = result.minimizer 


## ============================================ ##
        # hp-update (optimization) 

        # mean and covariance 
        mZero = MeanZero() ;            # zero mean function 
        kern  = SE( σ_f , l ) ;          # squared eponential kernel (hyperparams on log scale) 
        log_noise = σ_n ;              # (optional) log std dev of obs noise 

        # fit GP 
        y_train = dx - Θx*ξ   
        gp  = GP(t, y_train, mZero, kern, log_noise) 

        result = optimize!(gp) 
        σ_f = result.minimizer[1]
        l   = result.minimizer[2] 
        σ_n = result.minimizer[3] 
        
## ============================================ ##
        # z-update (soft thresholding) 
    
        # λ = f_hp( ξ, σ_f, l, σ_n )
        f_hp( ξ, σ_f, l, σ_n )

        λ = 0.1 

        z_old = z 
        ξ_hat = α*ξ + (1 .- α)*z_old 

        # z     = shrinkage(ξ_hat + u, λ/ρ) 
        κ  = λ/ρ 
        ξx = ξ_hat + u 
        z  = 0*ξx ; 
        for i = 1:length(ξx) 
            z[i] = max( 0, ξx[i] - κ ) - max( 0, -ξx[i] - κ ) 
        end 

        z_if = shrinkage( ξ_hat + u, λ/ρ )

        z 

## ============================================ ##
        # diagnostics + termination checks 

        # ----------------------- #
        # u-update 

        u += (ξ_hat - z) 


        p = f_hp(ξ, σ_f, l, σ_n) + g(z)   
        push!( hist.objval, p )
        push!( hist.fval, f_hp( ξ, σ_f, l, σ_n ) )
        push!( hist.gval, g(z) )
        push!( hist.hp, [ σ_f, l, σ_n ] )
        push!( hist.r_norm, norm(ξ - z) )
        push!( hist.s_norm, norm( -ρ*(z - z_old) ) )
        push!( hist.eps_pri, sqrt(n)*abstol + reltol*max(norm(ξ), norm(-z)) ) 
        push!( hist.eps_dual, sqrt(n)*abstol + reltol*norm(ρ*u) ) 

        # if hist.r_norm[k] < hist.eps_pri[k] && hist.s_norm[k] < hist.eps_dual[k] 
        #     break 
        # end 

    # end 
    
## ============================================ ##
# back to MONTE CARLO GP SINDy

    Ξ_sindy_err   = [ norm( Ξ_true[:,1] - Ξ_sindy[:,1] ), norm( Ξ_true[:,2] - Ξ_sindy[:,2] )  ] 
    z_gpsindy_err = [ norm( Ξ_true[:,1] - z_gpsindy[:,1] ), norm( Ξ_true[:,2] - z_gpsindy[:,2] )  ] 

    
