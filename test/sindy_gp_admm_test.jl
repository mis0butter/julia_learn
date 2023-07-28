using GaussianSINDy
using Statistics 

## ============================================ ##

noise_vec = [] 
noise_vec_iter = 0.05 : 0.05 : 0.3 
for i in noise_vec_iter 
    for j = 1:20 
        push!(noise_vec, i)
    end 
end 
# noise_vec = collect( 0 : 0.05 : 0.2 ) 
# noise_vec = 0.1 

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
case = 7 

λ = 0.1 
abstol = 1e-2 ; reltol = 1e-2           
sindy_err_vec, gpsindy_err_vec, hist_nvars_vec, Ξ_true, sindy_vec, gpsindy_vec = monte_carlo_gpsindy( noise_vec, λ, abstol, reltol, case ) 

println( "sindy err   = ", sindy_err_vec ) 
println( "gpsindy err = ", gpsindy_err_vec )  
println( "noise_vec = ", noise_vec ) 
println( "case = ", case )  


## ============================================ ##

n_vars   = size( sindy_err_vec, 2 ) 
unique_i = unique( i -> noise_vec[i], 1:length( noise_vec ) ) 
push!( unique_i, length(noise_vec)+1 ) 

sindy_med   = [] ; sindy_q13   = [] 
gpsindy_med = [] ; gpsindy_q13 = [] 
for i = 1 : length(unique_i)-1 

    ji = unique_i[i] 
    jf = unique_i[i+1]-1

    smed = [] ; gpsmed = [] ; sq13 = [] ; gpsq13 = [] 
    for j = 1 : n_vars 
        push!( smed,   median( sindy_err_vec[ji:jf, j] ) ) 
        push!( gpsmed, median( gpsindy_err_vec[ji:jf, j] ) ) 
        push!( sq13,   [ quantile( sindy_err_vec[ji:jf, j], 0.25 ), quantile( sindy_err_vec[ji:jf, j], 0.75 ) ] ) 
        push!( gpsq13,   [ quantile( gpsindy_err_vec[ji:jf, j], 0.25 ), quantile( gpsindy_err_vec[ji:jf, j], 0.75 ) ] ) 
    end 

    push!( sindy_med, smed )     ; push!( sindy_q13, sq13 ) 
    push!( gpsindy_med, gpsmed ) ; push!( gpsindy_q13, gpsq13 )  

end 
sindy_med   = vv2m(sindy_med)   ; sindy_q13   = vv2m(sindy_q13) 
gpsindy_med = vv2m(gpsindy_med) ; gpsindy_q13 = vv2m(gpsindy_q13)

using Plots 

p_nvars = [] 
for i = 1 : n_vars 
    plt = plot( legend = :outerright, size = [800 300], title = string("|| ξ", i, "_true - ξ", i, "_discovered ||") )
        ymed = sindy_med[:,i] ; yq13 = vv2m(sindy_q13[:,i])
        plot!( plt, noise_vec_iter, ymed, c = :orange, label = "SINDy", ribbon = (ymed - yq13[:,1], yq13[:,2] - ymed) ) 
        scatter!( plt, noise_vec, sindy_err_vec[:,i], c = :orange ) 
        ymed = gpsindy_med[:,i] ; yq13 = vv2m(gpsindy_q13[:,i])
        plot!( plt, noise_vec_iter, ymed, c = :cyan, label = "GPSINDy", ribbon = (ymed - yq13[:,1], yq13[:,2] - ymed) ) 
        scatter!( plt, noise_vec, gpsindy_err_vec[:,i], c = :cyan ) 
    push!( p_nvars, plt ) 
end 
p_nvars = plot( p_nvars ... ,  
    layout = (2,1), 
    size   = [800 600], 
    plot_title = "1/4, 1/2, and 3/4 Quartiles"
) 



## ============================================ ##
# boxplot plot 

boxplot_err( noise_vec, sindy_err_vec, gpsindy_err_vec )





