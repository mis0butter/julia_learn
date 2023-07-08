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
using Latexify

export plot_admm 
function plot_admm( hist, var ) 

    ind1 = 1
    ind2 = Int(round(1/2 * length(hist.objval), digits=1))
    if var == 2
        ind1 = ind2 + 1 
        ind2 = Int(length(hist.objval)) 
    end 

    K = length(hist.objval[ind1:ind2]) 
    
    # determine xtick range 

    # subplot 1 
    vals = hist.objval[ind1:ind2]
    ymin, dy, ymax = min_d_max( vals ) 
    println("vals length = ", length(vals)) 
    p_objval = plot( 1:K, vals, 
        legend = false , 
        title  = string( "\n Obj Fn = ", latexify( "f(x_k)+ g(z_k)" ) ), 
        yticks = ymin : dy : ymax , 
        ) 
    
    # subplot 2 
    vals1 = hist.r_norm[ind1:ind2] 
    ymin, dy, ymax = min_d_max( vals1 ) 
    vals2 = hist.eps_pri[ind1:ind2] 
    if maximum(vals2) > maximum(vals1) 
        ymin, dy, ymax = min_d_max( vals2 ) 
    end 
    p_r_norm = plot( 1:K, vals1, 
        title = string( "\n |r|", latexify("_2"), " = |x-z|", latexify("_2") ), 
        label = string( "|r|", latexify("_2") ),           
        yticks = ymin : dy : ymax , 
        ylim  = (ymin, ymax), 
        ) 
    plot!( p_r_norm, 1:K, vals2, 
        label = "tol", 
        ls = :dot, 
        )
    
    # subplot 3 
    vals1 = hist.s_norm[ind1:ind2] 
    ymin, dy, ymax = min_d_max( vals1 ) 
    vals2 = hist.eps_dual[ind1:ind2] 
    if maximum(vals2) > maximum(vals1) 
        ymin, dy, ymax = min_d_max( vals2 ) 
    end 
    p_s_norm = plot( 1:K, vals1, 
        title = string( "\n |s|", latexify("_2"), " = |-ρ(z-z", latexify("_old"), ")|", latexify("_2") ), 
        label = string( "|s|", latexify("_2") ),           
        yticks = ymin : dy : ymax , 
        ylim  = (ymin, ymax), 
        ) 
    plot!( p_s_norm, 1:K, vals2, 
        label = "tol", 
        ls = :dot, 
        )
    
    # plot all 
    p_fig = plot(p_objval, p_r_norm, 
        p_s_norm, layout = (1,3), size = [ 800,300 ],         
        plot_title  = string( "ADMM Lasso, ", latexify( "x_$(var)" ) ), 
        lw = 3, 
        xlabel = "iter", 
        # margin = 5Plots.mm,
        top_margin = 7Plots.mm, 
        legend = true, 
        xticks = 0:K/2:K, 
        )
        println("K = ", K)

    return p_fig 

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

    # update λ
    λ = f( x, σ_f, l, σ_n )
    
    # begin iterations 
    for k = 1 : max_iter 

        # increment counter 
        iter += 1 

        # ----------------------- # 
        # hp-update (optimization) 

        σ_0    = [σ_f, l, σ_n]  
        hp_opt(( σ_f, l, σ_n )) = aug_L(x, σ_f, l, σ_n, z, u) 
        od     = OnceDifferentiable( hp_opt, σ_0 ; autodiff = :forward ) 
        result = optimize( od, lower, upper, σ_0, Fminbox(LBFGS()) ) 
        
        # # assign optimized hyperparameters 
        σ_f = result.minimizer[1] 
        l   = result.minimizer[2] 
        σ_n = result.minimizer[3] 
        # σ_f = 1.0 
        # l   = 1.0 
        # σ_n = 0.1 

        # ----------------------- #
        # x-update (optimization) 

        # optimization 
        f_opt(x) = aug_L(x, σ_f, l, σ_n, z, u) 
        od       = OnceDifferentiable( f_opt, x ; autodiff = :forward ) 
        result   = optimize( od, x, LBFGS() ) 
        x        = result.minimizer 
        
        # ----------------------- #
        # z-update (soft thresholding) 
    
        λ = f( x, σ_f, l, σ_n )

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
        push!( hist.fval, f( x, σ_f, l, σ_n ) )
        push!( hist.gval, g(z) )
        push!( hist.hp, [ σ_f, l, σ_n ] )
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

