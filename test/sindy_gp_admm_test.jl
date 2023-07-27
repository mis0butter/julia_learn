using GaussianSINDy

## ============================================ ##

noise_vec = [] 
noise_vec_iter = 0.05 : 0.05 : 0.3 
for i in noise_vec_iter 
    for j = 1:100 
        push!(noise_vec, i)
    end 
end 
# noise_vec = collect( 0 : 0.05 : 0.2 )
# noise_vec = 0.0 

# case: 0 = true, 1 = finite difference, 2 = noise, 3 = standardize true, 4 = standardize noisy, 5 =  standardize and just use GP to smooth states 
case = 6 

λ = 0.1 
abstol = 1e-2 ; reltol = 1e-2           
sindy_err_vec, gpsindy_err_vec, hist_nvars_vec, Ξ_true, sindy_vec, gpsindy_vec = monte_carlo_gpsindy( noise_vec, λ, abstol, reltol, case ) 

println( "sindy err   = ", sindy_err_vec ) 
println( "gpsindy err = ", gpsindy_err_vec )  
println( "noise_vec = ", noise_vec ) 
println( "case = ", case )  


# ----------------------- #
# plot 

boxplot_err( noise_vec, sindy_err_vec, gpsindy_err_vec )





