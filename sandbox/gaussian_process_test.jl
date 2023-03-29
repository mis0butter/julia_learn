using Optim 
using LinearAlgebra
using Statistics 
using Plots 
using TickTock 
using GaussianProcesses 
using Random 
using Plots 
using ProgressMeter 
using BenchmarkTools
using ForwardDiff 

## ============================================ ##
## ============================================ ##
# functions 

# define square distance function 
function sq_dist2(a::Vector, b::Vector) 

    amat = repeat(a, 1, length(b))
    bmat = repeat(b', length(a), 1)
    Cmat = ( amat - bmat ).^2 

    return Cmat 

end 

# define square distance function 
function sq_dist(a::Vector, b::Vector) 

    r = length(a) ; 
    p = length(b) 

    # iterate 
    C = zeros(r,p) 
    for i = 1:r 
        for j = 1:p 
            C[i,j] = ( a[i] - b[j] )^2 
        end 
    end 

    return C 

end 

# test function 
a = rand(1000) 
b = rand(1000) 

println("efficient way? Cmat") 
@time C = sq_dist(b,a) ; 

display("for loops galore. C") 
@time C = sq_dist2(b,a) 

println("\n\n")


## ============================================ ##
# sample from given mean and covariance 
function gauss_sample(mu::Vector, K::Matrix) 
    
    # cholesky decomposition, get lower triangular decomp 
    C = cholesky(K) ; 
    L = C.L 

    # draw random samples 
    u = randn(length(mu)) 

    # f ~ N(mu, K(x, x)) 
    f = mu + L*u 

    return f 

end 

# test function 
C = rand(3,3)
K = C + C' + 10*I 
gauss_sample(rand(3), K)

## ============================================ ##
# kernel function 

function k_fn(( σ_f, l, xp, xq ))

    return σ_f^2 * exp.( -1/( 2*l^2 ) * sq_dist(xp, xq) ) 

end 

# test 
k_fn(( 1.0, 1.0, sort(rand(3)), sort(rand(3)) ))



## ============================================ ##
# marginal log-likelihood for Gaussian Process 

function log_p(( σ_f, l, σ_n, x, y, μ ))
    
    # kernel function 
    # k_fn(σ_f, l, xp, xq) = σ_f^2 * exp.( -1/( 2*l^2 ) * sq_dist(xp, xq) ) 

    # training kernel function 
    Ky = k_fn((σ_f, l, x, x)) + σ_n^2 * I 
    Ky += σ_n^2 * I 

    term_1 = 1/2 * y' * inv( Ky ) * y 
    term_2 = 1/2 * log( det( Ky ) ) 

    return term_1 + term_2

end 

# test log-likelihood function 
N = 10
x = sort(rand(N))
y = randn(N) 
μ = zeros(N) 

# test log_p 
log_p(( 1.0, 1.0, 0.1, x, y, μ ))

# test reassigning 
test_log_p( (σ_f, l, σ_n) ) = log_p( ( σ_f, l, σ_n, x, y, μ ) )

a = [1.0, 1.0, 0.1]
test_log_p(a)

ForwardDiff.gradient(test_log_p, a)


## ============================================ ##
# posterior distribution 

function post_dist(( x_train, y_train, x_test, σ_f, l, σ_n ))

    # x  = training data  
    # xs = test data 
    # joint distribution 
    #   [ y  ]     (    [ K(x,x)+Ïƒ_n^2*I  K(x,xs)  ] ) 
    #   [ fs ] ~ N ( 0, [ K(xs,x)         K(xs,xs) ] ) 

    # covariance from training data 
    K    = k_fn((σ_f, l, x_train, x_train))  
    K   += σ_n^2 * I       # add noise for positive definite 
    Ks   = k_fn((σ_f, l, x_train, x_test))  
    Kss  = k_fn((σ_f, l, x_test, x_test)) 

    # conditional distribution 
    # mu_cond    = K(Xs,X)*inv(K(X,X))*y
    # sigma_cond = K(Xs,Xs) - K(Xs,X)*inv(K(X,X))*K(X,Xs) 
    # fs | (Xs, X, y) ~ N ( mu_cond, sigma_cond ) 
    μ_post = Ks' * K^-1 * y_train 
    Σ_post = Kss - (Ks' * K^-1 * Ks)  

    return μ_post, Σ_post

end 


## ============================================ ##
## ============================================ ##
# create GP !!!  

Random.seed!(0) 

# true hyperparameters 
σ_f0 = 1.0 ;    σ_f = σ_f0 
l_0  = 1.0 ;    l   = l_0 
σ_n0 = 0.1 ;    σ_n = σ_n0 

# generate training data 
N = 100
x_train = sort( 2π*rand(N) ) 
N = length(x_train) 

# kernel function 
# k_fn(σ_f, l, xp, xq) = σ_f^2 * exp.( -1/( 2*l^2 ) * sq_dist(xp, xq) ) 

Σ_train = k_fn(( σ_f0, l_0, x_train, x_train ))
Σ_train += σ_n0^2 * I 

# training data --> "measured" output at x_train 
# y_train = gauss_sample( 0*x_train, Σ_train ) 
y_train = sin.(x_train) .+ 0.1*randn(N) 

# scatter plot of training data 
p_train = scatter(x_train, y_train, 
    c = :black, markersize = 5, label = "training points", markershape = :cross, title = "Fit GP", legend = :outerbottom ) 


## ============================================ ##
# posterior distribution ROUND 1 
# (based on training data) 
# NO hyperparameters tuned yet 

# test data 
x_test = collect( 0 : 0.1 : 2π )
Kss = k_fn(( σ_f0, l_0, x_test, x_test ))

# fit data 
μ_post, Σ_post = post_dist(( x_train, y_train, x_test, σ_f0, l_0, σ_n0 ))

# get covariances and stds 
cov_prior = diag(Kss );     std_prior = sqrt.(cov_prior); 
cov_post  = diag(Σ_post );  std_post  = sqrt.(cov_post); 

# plot fitted / predict / post data 
plot!(p_train, x_test, μ_post, rib = 3*std_post , lw = 3, fa = 0.15, c = :red, label = "μ ± 3σ (σ_0)")


# ## ============================================ ## 
# # solve for hyperparameters

println("samples = ", N) 

# test reassigning function 
test_log_p(( σ_f, l, σ_n )) = log_p(( σ_f, l, σ_n, x_train, y_train, 0*y_train )) 
test_log_p(( σ_f, l, σ_n )) 

σ_0   = [σ_f0, l_0, σ_n0]  
# σ_0    = [ σ_f, l_0, σ_n ] * 1.1 
lower = [0.0, 0.0, 0.0]  
upper = [Inf, Inf, Inf] 

# @time result = optimize( test_log_p, lower, upper, σ_0, Fminbox(LBFGS()) ) 
od = OnceDifferentiable( test_log_p, σ_0 ; autodiff = :forward ) 
@time result = optimize( od, lower, upper, σ_0, Fminbox(LBFGS()) ) 
println("log_p min (LBFGS) = \n ", result.minimizer) 

# assign optimized hyperparameters 
σ_f = result.minimizer[1] 
l   = result.minimizer[2] 
σ_n = result.minimizer[3] 


# ## ============================================ ##
# # optimize bounds test 

# f_test(x) = (x[1]-1)^2 + x[2]^2 
# x0     =  [2.0, 2.0] 
# # lower  = [-10.0, -10.0] 
# lower  = -[Inf, Inf]  
# upper  =  [Inf, Inf] 
# od     = OnceDifferentiable(f_test, x0; autodiff = :forward)
# result = optimize( od, lower, upper, x0, Fminbox(LBFGS()) ) 
# println("od = ", result.minimizer)


## ============================================ ##
# posterior distribution ROUND 2 
# (based on training data) 
# YES hyperparameters tuned 

# test data 
x_test = x_test 

μ_post, Σ_post = post_dist(( x_train, y_train, x_test, σ_f, l, σ_n ))

# get covariances and stds 
cov_prior = diag(Kss );     std_prior = sqrt.(cov_prior) 
cov_post  = diag(Σ_post );  std_post  = sqrt.(cov_post) 

# plot fitted / predict / post data 
plot!(p_train, x_test, μ_post, rib = 3*std_post , fillalpha = 0.15, c = :blue, label = "μ ± 3σ (σ_opt)")

xlabel!("x") 
ylabel!("y") 


## ============================================ ## 
# fit GP 

# mean and covariance 
mZero = MeanZero() ;                # zero mean function 
kern  = SE(σ_f0, l_0) ;             # squared eponential kernel (hyperparams on log scale) 
log_noise = log(σ_n0) ;             # (optional) log std dev of obs noise 

# fit GP 
gp = GP( x_train, y_train, mZero, kern, log_noise ) ; 
# optimize in a box with lower bounds [-1,-1] and upper bounds [1,1]
# optimize!(gp; kernbounds = [ [-1,-1] , [1,1] ])
p_gp_y = plot( gp; xlabel="x", ylabel="y", title="GP vs predict_y", fmt=:png ) ; 
p_gp_y_opt = plot( gp; xlabel="x", ylabel="y", title="GP vs predict_y (opt)", fmt=:png ) ; 

# predict at test points, should be same as gp plot?? 
μ_gp, σ²_gp = predict_y( gp, x_test ) 

# optimize gp 
test = GaussianProcesses.optimize!( gp; method = LBFGS() ) 
μ_gp_opt, σ²_gp_opt = predict_y( gp, x_test ) 

# "un-optimized" 
c = 3 ; 
plot!( p_gp_y, x_test, μ_gp, rib = 3*sqrt.(σ²_gp) , lw = 3, fillalpha = 0.15, c = c, label = "μ ± 3σ (predict_y)" )

# optimized 
c = 5 ; 
plot!( p_gp_y_opt, x_test, μ_gp_opt, rib = 3*sqrt.(σ²_gp_opt) , lw = 3, fillalpha = 0.15, c = c, label = "μ ± 3σ (predict_y, opt)" )

# plot un-optimized and optimized 
c = 3 ; 
p_y_y_opt = plot( x_test, μ_gp, c = c, rib = 3*sqrt.( σ²_gp ), lw = 3, fa = 0.15, label = "μ ± 3σ (predict_y)", title = "predict_y vs predict_y (opt)" )

c = 5 ; 
plot!( p_y_y_opt, x_test, μ_gp_opt, rib = 3*sqrt.(σ²_gp_opt) , lw = 3, fillalpha = 0.15, c = c, label = "μ ± 3σ (predict_y, opt)" )

# plot gp vs fitted 
p_gp_train = plot(gp; xlabel="x", ylabel="y", title="GP vs fitted", fmt=:png) 
c = 3 ; 

# plot fitted / predict / post data 
plot!( p_gp_train, x_test, μ_post, ribbon = 3*std_post , lw = 3, fillalpha = 0.35, c = 2, label = "μ ± 3σ (fitted)" ) 

# plot everything 
fig_gp_compare = plot( p_gp_y, p_gp_y_opt, p_gp_train, p_y_y_opt, layout = (4,1), size = [600 1000] )
display(fig_gp_compare) 


## ============================================ ##
# plot everything 

fig_gp_fit = plot( p_train, p_gp_train, layout = (2,1), size = [ 600, 800 ] )
display(fig_gp_fit) 


## ============================================ ##
# test 

# fa  = fillalpha, 
# lw  = linewidth, 
# lab = label, 
# rib = ribbon, 
# c   = color, 
# ms  = markersize 
# leg = legend 
# p_test = plot( x_test, μ_gp_opt, 
#     rib = 3 * sqrt.( σ²_gp_opt ), 
#     fa  = 0.15, 
#     c   = :green, 
#     lw  = 3, 
#     lab = "μ ± 3σ estimate" )