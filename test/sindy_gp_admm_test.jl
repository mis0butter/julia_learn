using GaussianSINDy
using Statistics 
using Random 

## ============================================ ##

# noise_vec = [] 
# noise_vec_iter = 0.05 : 0.01 : 0.3 
# for i in noise_vec_iter 
#     for j = 1:10 
#         push!(noise_vec, i)
#     end 
# end 
# noise_vec = collect( 0 : 0.05 : 0.2 ) 
noise_vec = 0.1  
fn = pendulum 

# ----------------------- #
# cases: 
# 0 = true, 
# 1 = finite difference, 
# 2 = noise, 
# 3 = stand x_true --> dx_true, 
# 4 = stand x_true --> dx_true, add noise, 
# 5 = stand x_true, dx_fd, 
# 6 = stand x_true --> dx_true, add noise, GP temporal smooth into SINDy, 
# 7 = stand x_true --> dx_true, add noise, GP NON-temporal smooth into SINDy, 
# 8 = stand x_true --> dx_true, add noise, GP NON-temporal smooth into GPSINDy
case = 9 

Random.seed!(1) 
λ = 0.1 
abstol = 1e-2 ; reltol = 1e-2           
sindy_err_vec, gpsindy_err_vec, hist_nvars_vec, Ξ_true, sindy_vec, gpsindy_vec = monte_carlo_gpsindy( fn, noise_vec, λ, abstol, reltol, case ) 

println( "sindy err   = ", sindy_err_vec ) 
println( "gpsindy err = ", gpsindy_err_vec )  
println( "noise_vec = ", noise_vec ) 
println( "case = ", case )  


## ============================================ ##
# plot monte carlo 

plot_med_quarts( sindy_err_vec, gpsindy_err_vec, noise_vec ) 
# boxplot_err( noise_vec, sindy_err_vec, gpsindy_err_vec )



## ============================================ ##







