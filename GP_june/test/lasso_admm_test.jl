struct Hist 
    objval 
    r_norm 
    s_norm 
    eps_pri 
    eps_dual 
end 


## ============================================ ##

using LinearAlgebra 
using GP_june 
using SparseArrays 
using Plots 
using Optim 


## ============================================ ##
## ============================================ ##
# setup 

m  = 15            # number of examples 
n  = 50            # number of features 
p  = 10/n           # sparsity density 
x0 = sprandn(n,1,p)    
A  = randn(m,n) 
b_rand = randn(m,1) 

# x0 = [ 1.16495351050066 , 0 , 0 ] 
# A  = [ 0.422443072642685    -0.144978181770844      0.999835350376907
#        0.906389458442786     0.989434852231525      0.0181458572872169 ] 
# b_rand = [  0.871673288690637 ,  -1.44617153933933 ] 
# n = length(x0) 

B  = 1 ./ sqrt.( sum(A.^2, dims=1) )
A  = A * spdiagm(n, n, B[:])            # normalize columns 
b  = A*x0 + sqrt(0.001) * b_rand 

λ  = 0.1 
ρ  = 1.0            # 1-norm threshold knob 
α  = 1.0            # over-relaxation parameter 
x  = rand(n,1) 
z  = rand(n,1) 

p = objective(A, b, λ, x, z) 
println("objective p = ", p)


## ============================================ ##
# lasso_admm 

x = zeros(n) 
z = zeros(n) 
u = zeros(n) 

f(x) = 1/2 * norm(A*x - b)^2 
g(z) = λ * sum(abs.(z)) 

hist_boyd = Hist( [], [], [], [], [] ) 
hist_opt  = Hist( [], [], [], [], [] ) 
hist_test = Hist( [], [], [], [], [] ) 

@time x_boyd, hist_boyd = lasso_admm_boyd(A, b, λ, ρ, α, hist_boyd) 
@time x_opt,  hist_opt  = lasso_admm_opt(f, g, n, λ, ρ, α, hist_opt) 
@time x_test, hist_test = lasso_admm_test( f, g, n, λ, ρ, α, hist_test ) 

println( "norm(x_boyd - x_opt) = ", norm(x_boyd - x_opt) )
println( "norm(x_boyd - x_test) = ", norm(x_boyd - x_test) )

## ============================================ ##
# plot! 

p_boyd = plot_admm(hist_boyd) 
    plot!(plot_title = "ADMM Lasso (Boyd)")
p_opt = plot_admm(hist_opt) 
    plot!(plot_title = "ADMM Lasso (x-opt)")
p_test = plot_admm(hist_test)
    plot!(plot_title = "ADMM Lasso (x-opt, z-opt)")

p_opt_test = plot(p_boyd, p_opt, p_test, layout = (1,3), size = [1200 1000])
    display(p_opt_test) 

# ## ============================================ ##
# ## ============================================ ##
# # sandbox 

#     # define constants 
#     max_iter = 1000  
#     abstol   = 1e-4 
#     reltol   = 1e-2 

#     # data pre-processing 
#     m, n = size(A) 
#     Atb = A'*b                          # save matrix-vector multiply 

#     # ADMM solver 
#     x = 0*Atb ; x0 = x 
#     z = 0*Atb 
#     u = 0*Atb 

#     # Optim stuff 
#     upper = Inf * ones(size(Atb)) 
#     lower = -upper 
#     f_test(x) = 1/2 * norm(A*x - b)^2 + ρ/2 .* norm(x - z + u)^2 

#     # cache factorization 
#     L, U = factor(A, ρ) 

#     k = 0 
    
# ## ============================================ ##
#     # for k = 1:max_iter 

#         k += 1 

#         # ----------------------- # 
#         # x-update 

#         q = Atb + ρ * (z - u)           # temp value 
#         if m >= n                       # if skinny 
#             x_lu = U \ ( L \ q ) 
#         else                            # if fat 
#             x_lu = q / ρ - ( A' * ( U \ ( L \ (A*q) ) ) ) / ρ^2 
#         end 

#         # inelegant way 
#         x_inv = inv( A'*A + ρ*I ) * ( Atb + ρ * ( z - u ) )

#         # optimization 
#         od     = OnceDifferentiable( f_test, x0 ; autodiff = :forward ) 
#         # result    = optimize( od, lower, upper, x0, Fminbox(LBFGS()) ) 
#         result = optimize( od, x0, LBFGS() ) 
#         x_opt  = result.minimizer 

#         println("norm(x_lu - x_inv) = ", norm(x_lu - x_inv))
#         println("norm(x_lu - x_opt) = ", norm(x_lu - x_opt)) 

#         x = x_opt 

#         # ----------------------- #
#         # z-update 

#         z_old = z 
#         x_hat = α*x + (1 .- α)*z_old 
#         z = shrinkage(x_hat + u, λ/ρ) 

#         # ----------------------- #
#         # u-update 

#         u = u + (x_hat - z) 

#         # ----------------------- #
#         # diagnostics + termination checks 

#         p = objective(A, b, λ, x, z) 
#         r_norm   = norm(x - z) 
#         s_norm   = norm( -ρ*(z - z_old) ) 
#         eps_pri  = sqrt(n)*abstol + reltol*max(norm(x), norm(-z)) 
#         eps_dual = sqrt(n)*abstol + reltol*norm(ρ*u)  

#         if hist.r_norm[k] < hist.eps_pri[k] && hist.s_norm[k] < hist.eps_dual[k] 
#             "reached tol!"
#         end 

#     # end 


# ## ============================================ ## 

#     f_test(x) = (x[1]-1)^2 + x[2]^2 
#     x0     =  [2.0, 2.0] 
#     # lower  = [-10.0, -10.0] 
#     lower  = -[Inf, Inf]  
#     upper  =  [Inf, Inf] 
#     od     = OnceDifferentiable(f_test, x0; autodiff = :forward)
#     # result = optimize( od, lower, upper, x0, Fminbox(LBFGS()) ) 
#     result = optimize( od, x0, LBFGS() ) 
#     println("min = ", result.minimizer)
    

# ## ============================================ ## 

# function f_test_wrap(f_test, x)

#     out = f_test(x) 

#     return out 

# end 

# f_test_wrap(f_test, x0) 

