
struct Hist 
    objval 
    fval 
    gval 
    hp 
    r_norm 
    s_norm 
    eps_pri 
    eps_dual 
end 

using GaussianSINDy
using LinearAlgebra 
using Plots 
using Dates 

## ============================================ ##
# choose ODE, plot states --> measurements 

#  
fn             = predator_prey 
plot_option    = 1 
savefig_option = 0 
fd_method      = 2 # 1 = forward, 2 = central, 3 = backward 

# choose ODE, plot states --> measurements 
x0, dt, t, x, dx_true, dx_fd = ode_states(fn, plot_option, fd_method) 

# truth coeffs 
Ξ_true = SINDy_test( x, dx_true, 0.1 ) 
Ξ_true = Ξ_true[:,1] 

dx_noise_vec = 0 : 0.1 : 1.0  
dx_noise = 0.0

Ξ_sindy_vec         = [] 
Ξ_sindy_err_vec     = [] 
z_gpsindy_vec       = [] 
z_gpsindy_err_vec   = [] 
hist_gpsindy_vec    = [] 
for dx_noise = dx_noise_vec 
    println("dx_noise = ", dx_noise)

    Ξ_sindy, Ξ_sindy_err, z_gpsindy, z_gpsindy_err, hist_gpsindy = monte_carlo_gpsindy( x0, dt, t, x, dx_true, dx_fd, dx_noise )

    println( "  Ξ_sindy = " );          println( Ξ_sindy )
    println( "  z_gpsindy = " );        println( z_gpsindy )
    println( "  Ξ_sindy_err = " );      println( Ξ_sindy_err )
    println( "  z_gpsindy_err = " );    println( z_gpsindy_err )

    push!(Ξ_sindy_vec, Ξ_sindy) 
    push!(z_gpsindy_vec, z_gpsindy)
    push!(Ξ_sindy_err_vec, Ξ_sindy_err) 
    push!(z_gpsindy_err_vec, z_gpsindy_err)
    push!( hist_gpsindy_vec, hist_gpsindy )
end 

Ξ_sindy_err_vec   = mapreduce(permutedims, vcat, Ξ_sindy_err_vec)
z_gpsindy_err_vec = mapreduce(permutedims, vcat, z_gpsindy_err_vec)


## ============================================ ##
# ----------------------- #
using Plots 

p_Ξ = [] 
for i = 1:2
    p_ξ = plot( dx_noise_vec, Ξ_sindy_err_vec[:,i], label = "SINDy" )
    plot!( p_ξ, dx_noise_vec, z_gpsindy_err_vec[:,i], ls = :dash, label = "GPSINDy" )
    plot!( p_ξ, 
        legend = false, 
        xlabel = "dx_noise", 
        title  = string( "ξ", i, "_true - ξ", i, "_discovered" ), 
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
    layout = grid( 1, 3, widths=[0.45, 0.45, 0.45] ) , 
    size   = [ 800 300 ], 
    ) 
display(p_Ξ)

t_str = string( "dx_true + dx_noise*randn \n dx_noise = ", minimum( dx_noise_vec ), " --> ", maximum( dx_noise_vec ) )
p_noise = plot( t, dx_true[:,1], title = t_str, xlabel = "Time (s)" )
for dx_noise = dx_noise_vec
    plot!(p_noise, t, dx_true[:,1] .+ dx_noise*randn( size(dx_true, 1), 1 ) ) 
end 
display(p_noise)

## ============================================ ##

obj_vals_fin = [] 
f_vals_fin   = [] 
g_vals_fin   = [] 
# plot last objvals 
for i in hist_gpsindy_vec 
    l = Int(length(i.objval)/2)
    push!( obj_vals_fin, [ i.objval[l], i.objval[end] ] )
    push!( f_vals_fin, [ i.fval[l], i.fval[end] ] )
    push!( g_vals_fin, [ i.gval[l], i.gval[end] ] )
end 
obj_vals_fin = vv2m( obj_vals_fin )
f_vals_fin   = vv2m( f_vals_fin )
g_vals_fin   = vv2m( g_vals_fin )

i = 1 
p_objvals = plot( dx_noise_vec, obj_vals_fin[:,i], label = "objval" )
    plot!( dx_noise_vec, f_vals_fin[:,i], ls = :dash, label = "fval" )
    plot!( dx_noise_vec, g_vals_fin[:,i], ls = :dot, label = "gval" )
    plot!( legend = true, title = "Final Fn Val for each iter", xlabel = "dx_noise" )

## ============================================ ##
# save data  

using JLD2

timestamp = Dates.format(now(), "YYYYmmdd-HHMMSS")
dir_name = joinpath(@__DIR__, "outputs", "runs_$timestamp")
@assert !ispath(dir_name) "Somebody else already created the directory"
if !ispath(dir_name)
    mkdir(dir_name)
end 

savefig(p_Ξ, string(dir_name, "\\p_Ξ.pdf") )
savefig(p_noise, string( dir_name, "\\p_noise_3nonlin.pdf" )) 

# save 
jldsave(string( dir_name, "\\batch_results_3nonlin.jld2" ); t, dx_noise_vec, dx_true, Ξ_sindy_vec, Ξ_sindy_err_vec, z_gpsindy_vec, z_gpsindy_err_vec, hist_gpsindy_vec)


## ============================================ ## 

