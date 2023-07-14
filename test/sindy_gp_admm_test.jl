using GaussianSINDy

## ============================================ ##

# noise_vec = [] 
# noise_vec_iter = 0 : 0.1 : 1.0 
# for i in noise_vec_iter 
#     for j = 1:10  
#         push!(noise_vec, i)
#     end 
# end 

noise_vec = collect( 0 : 0.1 : 0.1 )
# noise_vec = 0.0 

λ = 0.1 
abstol = 1e-2 ; reltol = 1e-2           
sindy_err_vec, gpsindy_err_vec, hist_nvars_vec = monte_carlo_gpsindy( noise_vec, λ, abstol, reltol ) 

## ============================================ ##

using Plots 
using StatsPlots 

p_Ξ = boxplot_err( noise_vec, sindy_err_vec, gpsindy_err_vec )


# t_str = string( "dx_true + dx_noise*randn \n dx_noise = ", minimum( noise_vec ), " --> ", maximum( noise_vec ) )
# p_noise = plot( t, dx_true[:,1], title = t_str, xlabel = "Time (s)" )
# for dx_noise = noise_vec
#     plot!(p_noise, t, dx_true[:,1] .+ dx_noise*randn( size(dx_true, 1), 1 ) ) 
# end 
# display(p_noise) 



