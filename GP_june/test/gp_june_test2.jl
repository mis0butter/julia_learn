using Optim 
using LinearAlgebra
using Statistics 
using Plots 
# using TickTock 
using GaussianProcesses 
using Random 
using GP_june 

## ============================================ ##
## ============================================ ##
# create GP !!!  

Random.seed!(0) 

# true hyperparameters 
σ_f0 = 1.0 ;    σ_f = σ_f0 
l_0  = 1.0 ;    l   = l_0 
σ_n0 = 0.1 ;    σ_n = σ_n0 

# generate training data 
N = 10
x_train = sort( 2π*rand(N) ) 
N = length(x_train) 

# kernel function 
k_fn(σ_f, l, xp, xq) = σ_f^2 * exp.( -1/( 2*l^2 ) * sq_dist(xp, xq) ) 

Σ_train = k_fn( σ_f0, l_0, x_train, x_train ) 
Σ_train += σ_n0^2 * I 

# training data --> "measured" output at x_train 
# y_train = gauss_sample( 0*x_train, Σ_train ) 
y_train = sin.(x_train) .+ 0.1*randn(N) 

p1 = scatter(x_train, y_train, 
    c = :black, markersize = 5, label = "training points", markershape = :cross, title = "Fit GP", legend = :outerbottom ) 


## ============================================ ##
# posterior distribution ROUND 1 
# (based on training data) 
# NO hyperparameters tuned yet 

# x  = training data  
# xs = test data 
# joint distribution 
#   [ y  ]     (    [ K(x,x)+Ïƒ_n^2*I  K(x,xs)  ] ) 
#   [ fs ] ~ N ( 0, [ K(xs,x)         K(xs,xs) ] ) 
x_test = collect( 0 : 0.1 : 2π )

# covariance from training data 
K    = k_fn(σ_f0, l_0, x_train, x_train)  
K   += σ_n0^2 * I       # add noise for positive definite 
Ks   = k_fn(σ_f0, l_0, x_train, x_test)  
Kss  = k_fn(σ_f0, l_0, x_test, x_test) 

# conditional distribution 
# mu_cond    = K(Xs,X)*inv(K(X,X))*y
# sigma_cond = K(Xs,Xs) - K(Xs,X)*inv(K(X,X))*K(X,Xs) 
# fs | (Xs, X, y) ~ N ( mu_cond, sigma_cond ); 
μ_post = Ks' * K^-1 * y_train ; 
Σ_post = Kss - (Ks' * K^-1 * Ks) ; 

# get covariances and stds 
cov_prior = diag(Kss );     std_prior = sqrt.(cov_prior); 
cov_post  = diag(Σ_post );  std_post  = sqrt.(cov_post); 

# plot fitted / predict / post data 
plot!(p1, x_test, μ_post, c = 1, label = "fitted mean (untrained) ")

# shade covariance 
plot!(p1, x_test, μ_post .- 3*std_post, fillrange = μ_post .+ 3*std_post , fillalpha = 0.35, c = 1, label = "3σ (untrained)")


## ============================================ ## 
# solve for hyperparameters

println("samples = ", N) 

# test reassigning function 
test_log_p(( σ_f, l, σ_n )) = log_p(( σ_f, l, σ_n, x_train, y_train, 0*y_train )) 
test_log_p(( σ_f, l, σ_n )) 

σ_0   = [σ_f0, l_0, σ_n0] * 1.1  
# σ_0    = [ σ_f, l_0, σ_n ] * 1.1 
lower = [0.0, 0.0, 0.0] 
upper = [Inf, Inf, Inf]

# result1 = optimize( test_log_p, σ_0, NelderMead() ) 
result3 = optimize( test_log_p, lower, upper, σ_0,  Fminbox(NelderMead()) ) 
println("log_p min (NelderMead) = \n ", result1.minimizer) 

# result2 = optimize( test_log_p, σ_0, BFGS() ) 
# println("log_p min (BFGS) = \n ", result2.minimizer) 

using Optim 

result3 = optimize( test_log_p, σ_0, LBFGS() ) 
# results = Optim.optimize(fmin, lower, upper, initial_x, Fminbox( LBFGS() ) )
result3 = optimize( test_log_p, lower, upper, σ_0,  Fminbox(LBFGS()) ) 
println("log_p min (LBFGS) = \n ", result3.minimizer) 

result = result1

# assign optimized hyperparameters 
σ_f = result.minimizer[1] 
l   = result.minimizer[2] 
σ_n = result.minimizer[3] 


## ============================================ ##
# posterior distribution ROUND 2 
# (based on training data) 
# YES hyperparameters tuned 

# x  = training data  
# xs = test data 
# joint distribution 
#   [ y  ]     (    [ K(x,x)+Ïƒ_n^2*I  K(x,xs)  ] ) 
#   [ fs ] ~ N ( 0, [ K(xs,x)         K(xs,xs) ] ) 
x_test = x_test 

# covariance from training data 
K    = k_fn(σ_f, l, x_train, x_train)   
K   +=  σ_n^2 * I       # add signal noise 
Ks   = k_fn(σ_f, l, x_train, x_test)  
Kss  = k_fn(σ_f, l, x_test, x_test) 

# conditional distribution 
# mu_cond    = K(Xs,X)*inv(K(X,X))*y
# sigma_cond = K(Xs,Xs) - K(Xs,X)*inv(K(X,X))*K(X,Xs) 
# fs | (Xs, X, y) ~ N ( mu_cond, sigma_cond ); 
μ_post = Ks' * K^-1 * y_train 
Σ_post = Kss - (Ks' * K^-1 * Ks)  

# get covariances and stds 
cov_prior = diag(Kss );     std_prior = sqrt.(cov_prior) 
cov_post  = diag(Σ_post );  std_post  = sqrt.(cov_post) 

# plot fitted / predict / post data 
plot!(p1, x_test, μ_post, c = 2, label = "fitted mean (trained) ")

# shade covariance 
plot!(p1, x_test, μ_post .- 3*std_post, fillrange = μ_post .+ 3*std_post , fillalpha = 0.35, c = 2, label = "3σ (trained)")

# create new plot 
p2 = scatter(x_train, y_train, 
    c = :black, markersize = 5, label = "training points", markershape = :cross, title = "Fit GP", legend = :outerbottom ) 

plot!(p2, x_test, μ_post, c = 2, label = "fitted mean (trained) ")
plot!(p2, x_test, μ_post .- 3*std_post, fillrange = μ_post .+ 3*std_post , fillalpha = 0.35, c = 2, label = "3σ (trained)")


## ============================================ ## 
# fit GP 

# mean and covariance 
mZero = MeanZero() ;            # zero mean function 
kern  = SE(σ_f0, l_0) ;          # squared eponential kernel (hyperparams on log scale) 
log_noise = log(σ_n0) ;              # (optional) log std dev of obs noise 

# fit GP 
gp  = GP(x_train, y_train, mZero, kern, log_noise) ; 
# optimize in a box with lower bounds [-1,-1] and upper bounds [1,1]
# optimize!(gp; kernbounds = [ [-1,-1] , [1,1] ])
p3 = plot(gp; xlabel="x", ylabel="y", title="GP vs predict_y", fmt=:png) ; 
p4 = plot(gp; xlabel="x", ylabel="y", title="GP vs predict_y (opt)", fmt=:png) ; 

# predict at test points, should be same as gp plot?? 
μ_gp, σ²_gp = predict_y( gp, x_test ) 

# optimize gp 
test = optimize!(gp; method = LBFGS() ) 
μ_gp_opt, σ²_gp_opt = predict_y( gp, x_test ) 

# "un-optimized" 
c = 3 ; 
plot!( p3, x_test, μ_gp, c = c, label = "mean (predict_y)" )
plot!( p3, x_test, μ_gp .- 3*sqrt.(σ²_gp), fillrange = μ .+ 3*sqrt.(σ²_gp) , fillalpha = 0.15, c = c, label = "3σ (predict_y)" )

# optimized 
c = 5 ; 
plot!( p4, x_test, μ_gp_opt, c = c, label = "mean (predict_y, opt)" )
plot!( p4, x_test, μ_gp_opt .- 3*sqrt.(σ²_gp_opt), fillrange = μ .+ 3*sqrt.(σ²_gp_opt) , fillalpha = 0.15, c = c, label = "3σ (predict_y, opt)" )

# plot un-optimized and optimized 
c = 3 ; 
p5 = plot( x_test, μ_gp, c = c, label = "mean (predict_y)", title = "predict_y vs predict_y (opt)" )
plot!( p5, x_test, μ_gp .- 3*sqrt.(σ²_gp), fillrange = μ .+ 3*sqrt.(σ²_gp) , fillalpha = 0.15, c = c, label = "3σ (predict_y)" )

c = 5 ; 
plot!( p5, x_test, μ_gp_opt, c = c, label = "mean (predict_y, opt)" )
plot!( p5, x_test, μ_gp_opt .- 3*sqrt.(σ²_gp_opt), fillrange = μ .+ 3*sqrt.(σ²_gp_opt) , fillalpha = 0.15, c = c, label = "3σ (predict_y, opt)" )


p6 = plot(gp; xlabel="x", ylabel="y", title="GP vs fitted", fmt=:png) 
c = 3 ; 

# plot fitted / predict / post data 
plot!(p6, x_test, μ_post, c = 2, label = "mean (fitted) ")

# shade covariance 
plot!(p6, x_test, μ_post .- 3*std_post, fillrange = μ_post .+ 3*std_post , fillalpha = 0.1, c = 2, label = "3σ (fitted)")

# plot everything 
p7 = plot( p3, p4, p6, p5, layout = (4,1), size = [600 1000] )

## ============================================ ##
# plot everything 

p8 = plot(p1, p2, layout = (2,1), size = [600 800] )

p6 = plot(gp; xlabel="x", ylabel="y", title="Gaussian Process", fmt=:png) 
c = 3 ; 

# plot fitted / predict / post data 
plot!(p6, x_test, μ_post, c = 2, label = "fitted mean (trained) ")

# shade covariance 
plot!(p6, x_test, μ_post .- 3*std_post, fillrange = μ_post .+ 3*std_post , fillalpha = 0.1, c = 2, label = "3σ (trained)")


## ============================================ ##
# test minimizing 1-norm 

# test_fn(z) = sum(abs.(z)) .+ z'*z 
# x = -10 : 0.1 : 10 
# x = collect(x) 

# y = 0*x 
# for i = 1:length(x) 
#     y[i] = test_fn(x[i])
# end 

# result = optimize(test_fn, 0.1) 
# println("minimizer = ", result.minimizer)

# plot(x,y) 