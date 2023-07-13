using GaussianSINDy

## ============================================ ##

# noise_vec = [] 
# noise_vec_iter = 0 : 0.1 : 1.0 
# for i in noise_vec_iter 
#     for j = 1:10  
#         push!(noise_vec, i)
#     end 
# end 

# noise_vec = collect( 0 : 0.1 : 0.1 )
noise_vec = 0.5 

λ = 0.1 
abstol = 1e-2 ; reltol = 1e-2           
sindy_err_vec, gpsindy_err_vec, hist_nvars_vec = monte_carlo_gpsindy( noise_vec, λ, abstol, reltol ) 

## ============================================ ##

using Plots 
using StatsPlots 

# noise_vec = 0 : 0.02 : 0.1 
# xmin, dx, xmax = min_d_max( noise_vec )

p_Ξ = [] 
for i = 1:2
    p_ξ = scatter( noise_vec, sindy_err_vec[:,i], shape = :circle, ms = 2, c = :blue, label = "SINDy" )
        boxplot!( p_ξ, noise_vec, sindy_err_vec[:,i], bar_width = 0.08, lw = 1, fillalpha = 0.2, c = :blue, linealpha = 0.5 )
        scatter!( p_ξ, noise_vec, gpsindy_err_vec[:,i], shape = :xcross, c = :red, label = "GPSINDy" )
        boxplot!( p_ξ, noise_vec, gpsindy_err_vec[:,i], bar_width = 0.04, lw = 1, fillalpha = 0.2, c = :red, linealpha = 0.5 ) 
        scatter!( p_ξ, 
            legend = false, 
            xlabel = "dx_noise", 
            title  = string( "|ξ", i, "_true - ξ", i, "_discovered|" ), 
            # xticks = xmin : dx : xmax, 
            ) 
    push!(p_Ξ, p_ξ)
end 
p = deepcopy(p_Ξ[end])  
plot!(p, 
    legend     = (-0.2,0.6) , 
    framestyle = :none , 
    title      = "", 
    )
push!( p_Ξ, p ) 
p_Ξ = plot(p_Ξ ... , 
    layout = grid( 1, 3, widths = [0.45, 0.45, 0.45] ) , 
    size   = [ 800 300 ], 
    ) 
display(p_Ξ)

# t_str = string( "dx_true + dx_noise*randn \n dx_noise = ", minimum( noise_vec ), " --> ", maximum( noise_vec ) )
# p_noise = plot( t, dx_true[:,1], title = t_str, xlabel = "Time (s)" )
# for dx_noise = noise_vec
#     plot!(p_noise, t, dx_true[:,1] .+ dx_noise*randn( size(dx_true, 1), 1 ) ) 
# end 
# display(p_noise) 



