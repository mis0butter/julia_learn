using GaussianSINDy

## ============================================ ##

# noise_vec = [] 
# noise_vec_iter = 0.05 : 0.05 : 0.1 
# for i in noise_vec_iter 
#     for j = 1:10  
#         push!(noise_vec, i)
#     end 
# end 
# noise_vec = collect( 0 : 0.05 : 0.2 )
noise_vec = 0.1  

# case: 0 = true, 1 = noise, 2 = standardize true, 3 = standardize noisy 
case = 3 

λ = 0.1 
abstol = 1e-2 ; reltol = 1e-2           
sindy_err_vec, gpsindy_err_vec, hist_nvars_vec = monte_carlo_gpsindy( noise_vec, λ, abstol, reltol, case ) 

println( "sindy err   = ", sindy_err_vec )
println( "gpsindy err = ", gpsindy_err_vec )  
println( "noise_vec = ", noise_vec )
println( "case = ", case )  


## ============================================ ##
# plot 

p_Ξ = boxplot_err( noise_vec, sindy_err_vec, gpsindy_err_vec )





