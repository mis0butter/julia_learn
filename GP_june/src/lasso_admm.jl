## ============================================ ##
# objective 

export objective 

function objective(A, b, λ, x, z) 

    p = ( 1/2 * sum( ( A*x - b ).^2 ) + λ*norm(z,1) ) 

    return p 
end 


## ============================================ ##
# shrinkage 

export shrinkage 

function shrinkage(x, κ) 

    z = 0*x ; 
    for i = 1:length(x) 
        z[i] = max( 0, x[i] - κ ) - max( 0, -x[i] - κ ) 
    end 

    return z 
end 


## ============================================ ##
# cache factorization 

export factor 

function factor(A, ρ)

    m, n =  size(A) ; 
    if m >= n 
        C = cholesky( A'*A + ρ*I ) 
    else
        C = cholesky( I + 1/ρ*(A*A') )  
    end 
    L = C.L  
    U = C.U 

    return L, U 
end 


## ============================================ ##
# plot everything! 

using Plots 
export plot_admm 

function plot_admm( hist ) 

    K = length(hist.objval) 

    # subplot 1 
    p_objval = plot( 1:K, hist.objval, 
        title = "Obj Fn = f(xₖ) + g(zₖ)", legend = false ) 
    
    # subplot 2 
    p_r_norm = plot( 1:K, hist.r_norm, 
        title = "|r|₂ = |x-z|₂", label = "|r|₂" ) 
    plot!( p_r_norm, 1:K, hist.eps_pri, 
        label = "tol", ls = :dot )
    
    # subplot 3 
    p_s_norm = plot(1:K, hist.s_norm, 
        title = "|s|₂ = |-ρ(z - z_old)|₂", label = "|s|₂" )
    plot!(p_s_norm, 1:K, hist.eps_dual, 
        label = "tol", ls = :dot )
    
    # plot all 
    p_fig = plot(p_objval, p_r_norm, 
        p_s_norm, layout = (3,1), size = [ 600,800 ], plot_title = "ADMM Lasso", lw = 2, xlabel = "iter" )

    return p_fig 

end 


## ============================================ ##
# LASSO ADMM! 

export lasso_admm_boyd 

function lasso_admm_boyd(A, b, λ, ρ, α, hist) 
# ----------------------- #
# lasso  Solve lasso problem via ADMM
#
# [z, history] = lasso(A, b, λ, ρ, α);
#
# Solves the following problem via ADMM:
#
#   minimize 1/2 || Ax - b ||₂² + λ || x ||₁
#
# The solution is returned in the vector x.
#
# history is a structure that contains:
#   objval   = objective function values 
#   r_norm   = primal residual norms 
#   s_norm   = dual residual norms 
#   eps_pri  = tolerances for the primal norms at each iteration
#   eps_dual = tolerance for dual residual norms at each iteration
#
# ρ is the augmented Lagrangian parameter.
#
# α is the over-relaxation parameter (typical values for α are
# between 1.0 and 1.8).
# 
# Reference: 
# http://www.stanford.edu/~boyd/papers/distr_opt_stat_learning_admm.html
# ----------------------- #

    # define constants 
    max_iter = 1000  
    abstol   = 1e-4 
    reltol   = 1e-2 

    # data pre-processing 
    m, n = size(A) 
    Atb = A'*b                          # save matrix-vector multiply 

    # ADMM solver 
    x = 0*Atb 
    z = 0*Atb 
    u = 0*Atb 

    # cache factorization 
    L, U = factor(A, ρ) 

    # begin iterations 
    for k = 1 : max_iter 

        # ----------------------- #
        # x-update 

        q = Atb + ρ * (z - u)           # temp value 
        if m >= n                       # if skinny 
            x = U \ ( L \ q ) 
        else                            # if fat 
            x = q / ρ - ( A' * ( U \ ( L \ (A*q) ) ) ) / ρ^2 
        end 

        # ----------------------- #
        # z-update 

        z_old = z 
        # x_hat = α*x + (1 .- α)*z_old 
        x_hat = x 
        z = shrinkage(x_hat + u, λ/ρ) 

        # ----------------------- #
        # u-update 

        u = u + (x_hat - z) 

        # ----------------------- #
        # diagnostics + termination checks 

        p = objective(A, b, λ, x, z) 
        push!( hist.objval, p )
        push!( hist.r_norm, norm(x - z) )
        push!( hist.s_norm, norm( -ρ*(z - z_old) ) )
        push!( hist.eps_pri, sqrt(n)*abstol + reltol*max(norm(x), norm(-z)) ) 
        push!( hist.eps_dual, sqrt(n)*abstol + reltol*norm(ρ*u) ) 

        if hist.r_norm[k] < hist.eps_pri[k] && hist.s_norm[k] < hist.eps_dual[k] 
            break 
        end 

    end 

    return x, z, hist 
end 
    

## ============================================ ##
# LASSO ADMM! 

using  Optim 
export lasso_admm_opt

function lasso_admm_opt( f, g, n, λ, ρ, α, hist ) 

    # define constants 
    max_iter = 1000  
    abstol   = 1e-4 
    reltol   = 1e-2           # save matrix-vector multiply 

    # ADMM solver 
    x = zeros(n) 
    z = zeros(n) 
    u = zeros(n) 

    # augmented Lagrangian (scaled form) 
    L(x, z, u) = f(x) + g(z) + ρ/2 .* norm( x - z + u )^2 

    # counter 
    iter = 0 
    
    # begin iterations 
    for k = 1 : max_iter 

        # increment counter 
        iter += 1 

        # ----------------------- #
        # x-update (optimization) 

        # optimization 
        f_opt(x) = L(x, z, u) 
        od       = OnceDifferentiable( f_opt, x ; autodiff = :forward ) 
        result   = optimize( od, x, LBFGS() ) 
        x        = result.minimizer 
        
        # ----------------------- #
        # z-update 

        z_old = z 
        x_hat = α*x + (1 .- α)*z_old 
        z = shrinkage(x_hat + u, λ/ρ) 

        # ----------------------- #
        # u-update 

        u = u + (x_hat - z) 

        # ----------------------- #
        # diagnostics + termination checks 

        p = f(x) + g(z)   
        push!( hist.objval, p )
        push!( hist.r_norm, norm(x - z) )
        push!( hist.s_norm, norm( -ρ*(z - z_old) ) )
        push!( hist.eps_pri, sqrt(n)*abstol + reltol*max(norm(x), norm(-z)) ) 
        push!( hist.eps_dual, sqrt(n)*abstol + reltol*norm(ρ*u) ) 

        if hist.r_norm[k] < hist.eps_pri[k] && hist.s_norm[k] < hist.eps_dual[k] 
            break 
        end 

    end 

    return x, z, hist, iter 

end 
    

## ============================================ ##
# LASSO ADMM! 

using  Optim 
export lasso_admm_test

function lasso_admm_test( f, g, n, λ, ρ, α, hist ) 

    # define constants 
    max_iter = 1000  
    abstol   = 1e-4 
    reltol   = 1e-2           # save matrix-vector multiply 

    # ADMM solver 
    x = zeros(n) 
    z = zeros(n) 
    u = zeros(n) 

    # augmented Lagrangian 
    L(x, z, u) = f(x) + g(z) + ρ/2 .* norm( x - z + u )^2 
    
    # begin iterations 
    for k = 1 : max_iter 

        # ----------------------- #
        # x-update (optimization) 

        f_opt(x) = L(x, z, u) 
        od       = OnceDifferentiable( f_opt, x ; autodiff = :forward ) 
        result   = optimize( od, x, LBFGS() ) 
        x        = result.minimizer 
        
        # ----------------------- #
        # z-update 

        # z_old = z 
        # x_hat = α*x + (1 .- α)*z_old 
        # z = shrinkage(x_hat + u, λ/ρ) 

        z_old    = z 
        g_opt(z) = L(x, z, u) 
        od       = OnceDifferentiable( g_opt, z ; autodiff = :forward ) 
        result   = optimize( od, z, LBFGS() ) 
        z        = result.minimizer 

        # ----------------------- #
        # u-update 

        u = u + (x - z) 

        # ----------------------- #
        # diagnostics + termination checks 

        p = f(x) + g(z)   
        push!( hist.objval, p )
        push!( hist.r_norm, norm(x - z) )
        push!( hist.s_norm, norm( -ρ*(z - z_old) ) )
        push!( hist.eps_pri, sqrt(n)*abstol + reltol*max(norm(x), norm(-z)) ) 
        push!( hist.eps_dual, sqrt(n)*abstol + reltol*norm(ρ*u) ) 

        if hist.r_norm[k] < hist.eps_pri[k] && hist.s_norm[k] < hist.eps_dual[k] 
            break 
        end 

    end 

    return x, z, hist

end 


    

## ============================================ ##
# LASSO ADMM! 

using  Optim 
export lasso_admm_hp_opt

function lasso_admm_hp_opt( f, g, n, λ, ρ, α, hist ) 

    # define constants 
    max_iter = 1000  
    abstol   = 1e-4 
    reltol   = 1e-2           # save matrix-vector multiply 

    # ADMM solver 
    x = z = u = zeros(n) 

    # initial hyperparameters 
    σ_f0 = 1.0 ; σ_f = σ_f0  
    l_0  = 1.0 ; l   = l_0   
    σ_n0 = 0.1 ; σ_n = σ_n0 

    # bounds 
    lower = [0.0, 0.0, 0.0]  
    upper = [Inf, Inf, Inf] 

    # augmented Lagrangian (scaled form) 
    aug_L(x, σ_f, l, σ_n, z, u) = f(x, σ_f, l, σ_n) + g(z) + ρ/2 .* norm( x - z + u )^2 

    # counter 
    iter = 0 
    
    # begin iterations 
    for k = 1 : max_iter 

        # increment counter 
        iter += 1 

        # ----------------------- #
        # x-update (optimization) 

        # optimization 
        f_opt(x) = aug_L(x, σ_f, l, σ_n, z, u) 
        od       = OnceDifferentiable( f_opt, x ; autodiff = :forward ) 
        result   = optimize( od, x, LBFGS() ) 
        x        = result.minimizer 

        # ----------------------- # 
        # hp-update (optimization) 

        σ_0    = [σ_f, l, σ_n]  
        hp_opt(( σ_f, l, σ_n )) = aug_L(x, σ_f, l, σ_n, z, u) 
        od     = OnceDifferentiable( hp_opt, σ_0 ; autodiff = :forward ) 
        result = optimize( od, lower, upper, σ_0, Fminbox(LBFGS()) ) 
        
        # assign optimized hyperparameters 
        σ_f = result.minimizer[1] 
        l   = result.minimizer[2] 
        σ_n = result.minimizer[3] 
        
        # ----------------------- #
        # z-update (soft thresholding) 

        z_old = z 
        x_hat = α*x + (1 .- α)*z_old 
        z     = shrinkage(x_hat + u, λ/ρ) 

        # ----------------------- #
        # u-update 

        u += (x_hat - z) 

        # ----------------------- #
        # diagnostics + termination checks 

        p = f(x, σ_f, l, σ_n) + g(z)   
        push!( hist.objval, p )
        push!( hist.r_norm, norm(x - z) )
        push!( hist.s_norm, norm( -ρ*(z - z_old) ) )
        push!( hist.eps_pri, sqrt(n)*abstol + reltol*max(norm(x), norm(-z)) ) 
        push!( hist.eps_dual, sqrt(n)*abstol + reltol*norm(ρ*u) ) 

        if hist.r_norm[k] < hist.eps_pri[k] && hist.s_norm[k] < hist.eps_dual[k] 
            break 
        end 

    end 

    return x, z, hist, iter 

end 


    
