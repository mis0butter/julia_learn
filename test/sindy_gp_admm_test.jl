using GaussianSINDy

## ============================================ ##

# noise_vec = [] 
# noise_vec_iter = 0 : 0.1 : 1.0 
# for i in noise_vec_iter 
#     for j = 1:10  
#         push!(noise_vec, i)
#     end 
# end 

# case: 0 = true, 1 = noise, 2 = normalize 
case = 1 

noise_vec = collect( 0 : 0.1 : 0.4 )
# noise_vec = 0.0 

λ = 0.1 
abstol = 1e-2 ; reltol = 1e-2           
sindy_err_vec, gpsindy_err_vec, hist_nvars_vec = monte_carlo_gpsindy( noise_vec, λ, abstol, reltol, case ) 

println( "sindy err   = ", sindy_err_vec )
println( "gpsindy err = ", gpsindy_err_vec )  

# ----------------------- #
# plot 

p_Ξ = boxplot_err( noise_vec, sindy_err_vec, gpsindy_err_vec )





