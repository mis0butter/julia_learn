using Optim 
using LinearAlgebra
using Statistics 
using Plots 
# using TickTock 
using GaussianProcesses 
using Random 
using Plots 
using ProgressMeter 
using BenchmarkTools
using ForwardDiff 
using GP_june 


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
y_train = sin.(x_train) .+ 0.1*randn(N) 

# training data covariance 
Σ_train = k_fn(( σ_f0, l_0, x_train, x_train ))
Σ_train += σ_n0^2 * I 

# scatter plot of training data 
p_train = scatter(x_train, y_train, 
    c = :black, markersize = 5, label = "training points", markershape = :cross, title = "Fit GP", legend = :outerbottom ) 

# test data points 
x_test = collect( 0 : 0.1 : 2π )

## ============================================ ##
# posterior distribution ROUND 1 (NO hyperparameters tuned yet)
# (based on training data) 

Kss = k_fn(( σ_f0, l_0, x_test, x_test ))

# fit data 
μ_post, Σ_post = post_dist(( x_train, y_train, x_test, σ_f0, l_0, σ_n0 ))

# get covariances and stds 
cov_prior = diag(Kss );     std_prior = sqrt.(cov_prior); 
cov_post  = diag(Σ_post );  std_post  = sqrt.(cov_post); 

# plot fitted / predict / post data 
plot!(p_train, x_test, μ_post, rib = 3*std_post , lw = 3, fa = 0.15, c = :red, label = "μ ± 3σ (σ_0)")


## ============================================ ## 
# solve for hyperparameters

println("samples = ", N) 

# test reassigning function 
log_p_hp(( σ_f, l, σ_n )) = log_p(( σ_f, l, σ_n, x_train, y_train, 0*y_train )) 
log_p_hp(( σ_f, l, σ_n )) 

σ_0   = [σ_f0, l_0, σ_n0]  
lower = [0.0, 0.0, 0.0]  
upper = [Inf, Inf, Inf] 

# @time result = optimize( test_log_p, lower, upper, σ_0, Fminbox(LBFGS()) ) 
od = OnceDifferentiable( log_p_hp, σ_0 ; autodiff = :forward ) 
@time result = optimize( od, lower, upper, σ_0, Fminbox(LBFGS()) ) 
println("log_p min (LBFGS) = \n ", result.minimizer) 

# assign optimized hyperparameters 
σ_f = result.minimizer[1] 
l   = result.minimizer[2] 
σ_n = result.minimizer[3] 


## ============================================ ##
# posterior distribution ROUND 2 (YES hyperparameters tuned)
# (based on training data) 

μ_post, Σ_post = post_dist(( x_train, y_train, x_test, σ_f, l, σ_n ))

# get covariances and stds 
cov_prior = diag(Kss );     std_prior = sqrt.(cov_prior) 
cov_post  = diag(Σ_post );  std_post  = sqrt.(cov_post) 

# plot fitted / predict / post data 
plot!(p_train, x_test, μ_post, rib = 3*std_post, fillalpha = 0.15, c = :blue, label = "μ ± 3σ (σ_opt)")

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
