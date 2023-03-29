

using LinearAlgebra 
using GP_june 
using SparseArrays
using Plots 
using Optim 

## ============================================ ##
# setup 

m = 2            # number of examples 
n = 3            # number of features 
p = 1/n           # sparsity density 

x0 = sprandn(n,1,p)    
A  = randn(m,n) 
A  = [  0.422443072642685        -0.144978181770844         0.999835350376907
        0.906389458442786         0.989434852231525        0.0181458572872169 ] 

B  = 1 ./ sqrt.( sum(A.^2, dims=1) )
A  = A * spdiagm(n, n, B[:])
b  = A*x0 + sqrt(0.001) * randn(m,1) 

λ  = 0.1 
ρ  = 1.0            # 1-norm threshold knob 
α  = 1.0            # over-relaxation parameter 
x  = rand(n,1) 
z  = rand(n,1) 

p = objective(A, b, λ, x, z) 
println("objective p = ", p)

## ============================================ ##
# lasso_admm 
    
x, hist = lasso_admm(A, b, λ, ρ, α) 

# x2, hist2 = lasso_admm2(A, b, λ, λ, ρ) 

## ============================================ ##
# plot! 

K = length(hist.objval) 

p_objval = plot(1:K, hist.objval, title = "Objective Function = f(xₖ) + g(zₖ)", legend = false) 
p_r_norm = plot(1:K, hist.r_norm, title = "Primal variables |r|₂ = |x-z|₂", label = "|r|₂") 
    plot!(p_r_norm, 1:K, hist.eps_pri, label = "tol")
p_s_norm = plot(1:K, hist.s_norm, title = "Dual variables |s|₂ = |-ρ(z - z_old)|₂", label = "|s|₂")
    plot!(p_s_norm, 1:K, hist.eps_dual, label = "tol")
p_fig = plot(p_objval, p_r_norm, p_s_norm, layout = (3,1), size = [ 600,800 ], plot_title = "test" )


## ============================================ ##
# sandbox 

    # define constants 
    max_iter = 1000  
    abstol   = 1e-4 
    reltol   = 1e-2 

    # data pre-processing 
    m, n = size(A) 
    Atb = A'*b                          # save matrix-vector multiply 

    # ADMM solver 
    x = 0*Atb ; x0 = x 
    z = 0*Atb 
    u = 0*Atb 

    # Optim stuff 
    upper = Inf * ones(size(Atb)) 
    lower = -upper 
    f_test(x) = 1/2 * norm(A*x - b)^2 + ρ/2 .* norm(x - z + u)^2 

    # cache factorization 
    L, U = factor(A, ρ) 
    
    # for k = 1:max_iter 

        # x-update 
        q = Atb + ρ * (z - u)           # temp value 
        if m >= n                       # if skinny 
            x = U \ ( L \ q ) 
        else                            # if fat 
            x = q / ρ - ( A' * ( U \ ( L \ (A*q) ) ) ) / ρ^2 
        end 

        # inelegant way 
        x_inv = inv( A'*A + ρ*I ) * ( Atb + ρ * ( z - u ) )

        # optimization 
        od        = OnceDifferentiable( f_test, x0 ; autodiff = :forward ) 
        # result    = optimize( od, lower, upper, x0, Fminbox(LBFGS()) ) 
        result    = optimize( od, x0, LBFGS() ) 
        x_opt = result.minimizer 

        println("norm(x - x_inv) = ", norm(x - x_inv))
        println("norm(x - x_opt) = ", norm(x - x_opt))

    # end 

## ============================================ ##

    f_test(x) = (x[1]-1)^2 + x[2]^2 
    x0     =  [2.0, 2.0] 
    # lower  = [-10.0, -10.0] 
    lower  = -[Inf, Inf]  
    upper  =  [Inf, Inf] 
    od     = OnceDifferentiable(f_test, x0; autodiff = :forward)
    # result = optimize( od, lower, upper, x0, Fminbox(LBFGS()) ) 
    result = optimize( od, x0, LBFGS() ) 
    println("od = ", result.minimizer)
    
