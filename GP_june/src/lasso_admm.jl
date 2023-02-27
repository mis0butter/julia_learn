struct Hist 
    objval 
    r_norm 
    s_norm 
    eps_pri 
    eps_dual 
end 

## ============================================ ##

export lasso_admm 
export factor 
export shrinkage 
export objective 

## ============================================ ##

# objective 
function objective(A, b, lambda, x, z) 

    p = ( 1/2 * sum( ( A*x - b ).^2 ) + lambda*norm(z,1) ) 

    return p 
end 

# shrinkage 
function shrinkage(x, kappa) 

    z = 0*x ; 
    for i = 1:length(x) 
        z[i] = max( 0, x[i] - kappa ) - max( 0, -x[i] - kappa ) 
    end 

    return z 
end 

# cache factorization 
function factor(A, rho)

    m, n =  size(A) ; 
    if m >= n 
        C = cholesky( A'*A + rho*I ) 
    else
        C = cholesky( I + 1/rho*(A*A') )  
    end 
    L = C.L  
    U = C.U 

    return L, U 
end 

# end 

function lasso_admm(A, b, lamda, rho, alpha) 
    # ------------------------------------------------------------------------
    # lasso  Solve lasso problem via ADMM
    #
    # [z, history] = lasso(A, b, lambda, rho, alpha);
    #
    # Solves the following problem via ADMM:
    #
    #   minimize 1/2*|| Ax - b ||_2^2 + \lambda || x ||_1
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
    # rho is the augmented Lagrangian parameter.
    #
    # alpha is the over-relaxation parameter (typical values for alpha are
    # between 1.0 and 1.8).
    # 
    # Reference: 
    # http://www.stanford.edu/~boyd/papers/distr_opt_stat_learning_admm.html
    # ------------------------------------------------------------------------
    
        # define constants 
        max_iter = 1000  
        abstol   = 1e-4 
        reltol   = 1e-2 
    
        # data pre-processing 
        m, n = size(A) 
        Atb = A'*b                          # save matrix-vector multiply 
    
        # ADMM solver 
        x = 0*b  
        z = 0*b 
        u = 0*b 
    
        # cache factorization 
        L, U = factor(A, rho) 
    
        # begin iterations 
        for k = 1:max_iter 
    
            # x-update 
            q = Atb + rho*(z .- u)           # temp value 
            if m >= n                       # if skinny 
                x = U \ ( L \ q ) 
            else                            # if fat 
                x = q / rho - ( A' * ( U \ ( L \ (A*q) ) ) ) / rho^2 
            end 
    
            # z-update 
            z_old = z 
            x_hat = alpha*x + (1 .- alpha*z_old) 
            z = shrinkage(x_hat + u, lambda/rho) 
    
            # u-update 
            u = u + (x_hat .- z) 
    
            # diagnostics + termination checks 
            p = objective(A, b, lambda, x, z) 
            push!( hist.objval, p )
            push!( hist.r_norm, norm(x - z) )
            push!( hist.s_norm, norm( -rho*(z - z_old) ) )
            push!( hist.eps_pri, sqrt(n)*abstol + reltol*max(norm(x), norm(-z)) ) 
            push!( hist.eps_dual, sqrt(n)*abstol + reltol*norm(rho*u) ) 
    
            if hist.r_norm[k] < hist.eps_pri[k] && hist.s_norm[k] < hist.eps_dual[k] 
                break 
            end 
    
        end 
    
        return z, hist 
    end 
    
    