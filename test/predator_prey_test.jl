struct Hist 
    objval 
    r_norm 
    s_norm 
    eps_pri 
    eps_dual 
end 

using DifferentialEquations 
using GaussianSINDy
using LinearAlgebra 
using ForwardDiff 
using Optim 
using Plots 
using CSV 
using DataFrames 
using Symbolics 
using PrettyTables 
using Test 
using NoiseRobustDifferentiation
using Random, Distributions 

using LaTeXStrings
using Latexify


## ============================================ ##
# choose ODE, plot states --> measurements 

#  
fn          = predator_prey 
plot_option = 1 
t, x, dx_true, dx_fd = ode_states(fn, plot_option) 

# split into training and validation data 
train_fraction = 0.7 
t_train, t_test             = split_train_test(t, train_fraction) 
x_train, x_test             = split_train_test(x, train_fraction) 
dx_true_train, dx_true_test = split_train_test(dx_true, train_fraction) 
dx_fd_train, dx_fd_test     = split_train_test(dx_fd, train_fraction) 


## ============================================ ##
# SINDy alone 

λ = 0.1  
n_vars     = size(x, 2) 
poly_order = n_vars 

Ξ_true = SINDy( x_train, dx_true_train, λ )
Ξ_sindy_fd   = SINDy( x_train, dx_fd_train, λ )


## ============================================ ##
# SINDy + GP + ADMM 

# # truth 
# hist_true = Hist( [], [], [], [], [] ) 
# @time z_true, hist_true = sindy_gp_admm( x, dx_true, λ, hist_true ) 
# display(z_true) 

λ = 2e-2 
println("λ = ", λ) 

# finite difference 
hist_fd = Hist( [], [], [], [], [] ) 
@time Ξ_gpsindy_fd, hist_fd = sindy_gp_admm( x_train, dx_fd_train, λ, hist_fd ) 
display(Ξ_gpsindy_fd) 


## ============================================ ## 
# generate + validate data 

dx_gpsindy_fn = build_dx_fn(poly_order, Ξ_sindy_fd) 
dx_sindy_fn   = build_dx_fn(poly_order, Ξ_gpsindy_fd)

t_gpsindy_val, x_gpsindy_val = validate_data(t_test, x_test, dx_gpsindy_fn)
t_sindy_val, x_sindy_val     = validate_data(t_test, x_test, dx_sindy_fn)

# compute metrics 
norm_sindy_val = zeros(size(x_test,2))
norm_gpsindy_val = zeros(size(x_test,2))
for i = 1:size(x_test,2) 
    norm_sindy_val[i]   = norm( x_test[:,i] - x_sindy_val[:,i] )
    norm_gpsindy_val[i] = norm( x_test[:,i] - x_gpsindy_val[:,i] )
end 

## ============================================ ##
# plots 

plot_font = "Computer Modern" 
default(
    fontfamily = plot_font,
    linewidth = 2, 
    # framestyle = :box, 
    label = nothing, 
    grid = false, 
    )
# scalefontsizes(1.1)
ptitles = ["Prey", "Predator"]
fsize = 18 

plot_vec = [] 
for i = 1:n_vars 

    # display training data 
    p = plot(t_train, x_train[:,i], 
        margin = 10Plots.mm, 
        lw = 3, 
        c = :gray, 
        label = "train (70%)", 
        grid = false, 
        xlim = (t_train[end]*3/4, t_test[end]), 
        legend = :bottomleft , 
        # legend = true , 
        xlabel = "Time (s)", 
        # title  = string(ptitles[i],  latexify(", x_$(i)")  ), 
        # title  = string( ptitles[i], latexify( string(" x ", ", x_$(i)") ) ), 
        title  = string( ptitles[i], ", ", latexify( "x_$(i)" ) ), 
        xticks = 0:2:10, 
        yticks = 0:0.5:4,   
        tickfontsize = fsize, 
        legendfontsize = fsize-2, 
        xguidefontsize = fsize, 
        yguidefontsize = fsize, 
        titlefontsize = fsize, 
        ) 
    plot!(t_test, x_test[:,i], 
        ls = :dash, 
        c = :blue,
        lw = 3,  
        label = "test (30%)" 
        )
        plot!(t_sindy_val, x_sindy_val[:,i], 
            ls = :dashdot, 
            lw = 1.5, 
            c = :green, 
            label = "SINDy" 
            )
    plot!(t_gpsindy_val, x_gpsindy_val[:,i], 
        ls = :dash, 
        lw = 1.5, 
        c = :red, 
        label = "GP SINDy" 
        )

    push!( plot_vec, p ) 

end 

plot!(legend = false)

p_train_val = plot(plot_vec ... , 
    layout = (1, n_vars), 
    size = [ n_vars*600 400 ], 
    # plot_title = "Training vs. Validation Data", 
    # titlefont = font(16), 
    )
display(p_train_val) 


## ============================================ ##


savefig("./predator_prey.pdf")


