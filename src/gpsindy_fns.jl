using Optim 
using GaussianProcesses
using LinearAlgebra 
using Statistics 
using Plots 

## ============================================ ##

export admm_lasso 
function admm_lasso( t, dx, x, (ξ, z, u), hp, λ, α, ρ, abstol, reltol, hist ) 
# ----------------------- #
# PURPOSE: 
#       Run one iteration of ADMM LASSO 
# INPUTS: 
#       t           : training data ordinates time  
#       dx          : training data ( f(x) )
#       x           : training data ( x ) 
#       (ξ, z, u)   : dynamics coefficients primary and dual vars 
#       hp          : hyperparameters 
#       λ           : L1 norm threshold  
#       α           : relaxation parameter 
#       ρ           : idk what this does, but Boyd sets it to 1 
#       print_vars  : option to display ξ, z, u, hp 
#       abstol      : abs tol 
#       reltol      : rel tol 
#       hist        : diagnostics struct 
# OUTPUTS: 
#       ξ           : output dynamics coefficients (ADMM primary variable x)
#       z           : output dynamics coefficients (ADMM primary variable z)
#       u           : input dual variable 
#       hp          : log-scaled hyperparameters 
#       hist        : diagnostis struct 
#       plt         : plot for checking performance of coefficients 
# ----------------------- #

    n_vars = size(x, 2) ; poly_order = n_vars 
    Θx     = pool_data_test( x, n_vars, poly_order ) 

    # objective fns 
    f_hp, g, aug_L = obj_fns( t, dx, x, λ, ρ )

    # hp-update (optimization) 
    # hp = opt_hp(t, dx, x, ξ) 

    # ξ-update 
    ξ = opt_ξ( aug_L, ξ, z, u, hp ) 
    
    # z-update (soft thresholding) 
    z_old = z 
    ξ_hat = α*ξ + (1 .- α)*z_old 
    z     = shrinkage( ξ_hat + u, λ/ρ ) 

    # u-update 
    u += (ξ_hat - z) 
    
    # ----------------------- #
    # plot 
    plt = scatter( t, dx, label = "train (noise)", c = :black, ms = 3 ) 
    plot!( plt, t, Θx*ξ, label = "Θx*ξ", c = :green, ms = 3 ) 

    # ----------------------- #
    # push diagnostics 
    n = length(ξ) 
    push!( hist.objval, f_hp(ξ, hp) + g(z) )
    push!( hist.fval, f_hp( ξ, hp ) )
    push!( hist.gval, g(z) ) 
    push!( hist.hp, hp )
    push!( hist.r_norm, norm(ξ - z) )
    push!( hist.s_norm, norm( -ρ*(z - z_old) ) )
    push!( hist.eps_pri, sqrt(n)*abstol + reltol*max(norm(ξ), norm(-z)) ) 
    push!( hist.eps_dual, sqrt(n)*abstol + reltol*norm(ρ*u) ) 
    
    return ξ, z, u, hp, hist, plt 
end 

## ============================================ ##

export gpsindy 
function gpsindy( t, dx_train, x, λ, α, ρ, abstol, reltol ) 
# ----------------------- # 
# PURPOSE: 
#       Main gpsindy function (iterate j = 1 : n_vars) 
# INPUTS: 
#       t       : training data ordinates ( x ) 
#       dx      : training data ( f(x) )
#       x       : training data ( x )  
#       λ       : L1 norm threshold  
#       α       : relaxation parameter 
#       ρ       : idk what this does, but Boyd sets it to 1 
#       abstol  : abs tol 
#       reltol  : rel tol 
# OUTPUTS: 
#       Ξ       : sparse dynamics coefficient (hopefully) 
#       hist    : diagnostics struct 
# ----------------------- # 

# set up 
n_vars = size(dx_train, 2) ; poly_order = n_vars 
Θx     = pool_data_test( x, n_vars, poly_order ) 
Ξ      = zeros( size(Θx, 2), size(dx_train, 2) ) 
hist_nvars = [] 
for j = 1 : n_vars 

    dx = dx_train[:,j] 
    
    # start animation 
    # a = Animation() 
    # plt = scatter( t, dx, label = "train (noise)", c = :black, ms = 3 ) 
    # plot!( plt, legend = :outerright, size = [800 300], title = string("Fitting ξ", j), xlabel = "Time (s)" ) 
    # frame(a, plt) 

    # ξ-update 
    n = size(Θx, 2); ξ = z = u = zeros(n) 
    f_hp, g, aug_L = obj_fns( t, dx, x, λ, ρ )
    hp = [1.0, 1.0, 0.1] 
    ξ  = opt_ξ( aug_L, ξ, z, u, hp ) 
    # dx_GP, Σ_dxsmooth, hp = post_dist_SE( x, x, dx )  
    println( "hp = ", hp ) 

    hist = Hist( [], [], [], [], [], [], [], [] )  

    # ----------------------- #
    # hp = opt_hp(t, dx, x, ξ) 
    # loop until convergence or max iter 
    for k = 1 : 1000  

        # ADMM LASSO! 
        z_old = z 
        ξ, z, u, hp, hist, plt = admm_lasso( t, dx, x, (ξ, z, u), hp, λ, α, ρ, abstol, reltol, hist )    

        # end condition 
        # if hist.r_norm[end] < hist.eps_pri[end] && hist.s_norm[end] < hist.eps_dual[end] 
        if hist.r_norm[end] < abstol[end] && hist.s_norm[end] < abstol[end] 
            break 
        end 

    end 
    # plot!( plt, legend = :outerright, size = [800 300], title = string("Fitting ξ", j), xlabel = "Time (s)" )  
    # frame(a, plt) 

    # ----------------------- #
    # optimize HPs 
    hp = opt_hp( t, dx, x, ξ ) 
    # loop until convergence or max iter 
    for k = 1 : 1000  

        # ADMM LASSO! 
        z_old = z 
        ξ, z, u, hp, hist, plt = admm_lasso( t, dx, x, (ξ, z, u), hp, λ, α, ρ, abstol, reltol, hist )    

        # end condition 
        # if hist.r_norm[end] < hist.eps_pri[end] && hist.s_norm[end] < hist.eps_dual[end] 
        if hist.r_norm[end] < abstol[end] && hist.s_norm[end] < abstol[end] 
            break 
        end 

    end 
    # plot!( plt, legend = :outerright, size = [800 300], title = string("Fitting ξ", j), xlabel = "Time (s)" )  
    # frame(a, plt) 

    # ----------------------- #
    # optimize HPs 
    hp = opt_hp(t, dx, x, ξ) 
    # loop until convergence or max iter 
    for k = 1 : 1000  

        # ADMM LASSO! 
        z_old = z 
        ξ, z, u, hp, hist, plt = admm_lasso( t, dx, x, (ξ, z, u), hp, λ, α, ρ, abstol, reltol, hist )    

        # end condition 
        # if hist.r_norm[end] < hist.eps_pri[end] && hist.s_norm[end] < hist.eps_dual[end] 
        if hist.r_norm[end] < abstol[end] && hist.s_norm[end] < abstol[end] 
            break 
        end 

    end 
    # plot!( plt, legend = :outerright, size = [800 300], title = string("Fitting ξ", j), xlabel = "Time (s)" )  
    # frame(a, plt) 

    # push diagnostics 
    push!( hist_nvars, hist ) 
    Ξ[:,j] = z 

    # g = gif(a, fps = 2) 
    # display(g) 
    # display(plt) 
    
    end 

    return Ξ, hist_nvars 
end 

## ============================================ ##

export monte_carlo_gpsindy 
function monte_carlo_gpsindy( noise_vec, λ, abstol, reltol, case ) 
# ----------------------- #
# PURPOSE:  
#       Run GPSINDy monte carlo 
# INPUTS: 
#       noise_vec       : dx noise vector for iterations 
#       λ               : L1 norm threshold 
#       abstol          : abs tol 
#       reltol          : rel tol 
#       case            : 0 = true, 1 = noise, 2 = norm 
# OUTPUTS: 
#       sindy_err_vec   : sindy error stats 
#       gpsindy_err_vec : gpsindy error stats 
# ----------------------- # 
    
    # choose ODE, plot states --> measurements 
    fn = predator_prey 
    x0, dt, t, x_true, dx_true, dx_fd, p = ode_states(fn, 0, 2) 
    
    # truth coeffs 
    n_vars = size(x_true, 2) ; poly_order = n_vars 
    Ξ_true = SINDy_test( x_true, dx_true, λ ) 

    # constants 
    α = 1.0  ; ρ = 1.0     

    sindy_err_vec = [] ; gpsindy_err_vec = [] ; hist_nvars_vec = [] 
    sindy_vec = [] ; gpsindy_vec = [] 
    for noise = noise_vec 
    
        # use true data 
        if case == 0 

            # set noise = true 
            x_noise  = x_true ; dx_noise = dx_true 
            
            Θx_sindy = pool_data_test( x_noise, n_vars, poly_order ) 
            Ξ_sindy  = SINDy_test( x_noise, dx_noise, λ ) 
            Θx_gpsindy = pool_data_test(x_noise, n_vars, poly_order) 
            Ξ_gpsindy, hist_nvars = gpsindy( t, dx_noise, x_noise, λ, α, ρ, abstol, reltol )  
            # Ξ_gpsindy = Ξ_sindy ; hist_nvars = [] 
    
        # use true x, finite difference dx 
        elseif case == 1 
    
            # set noise = true 
            x_noise  = x_true ; dx_noise = dx_fd 
            
            Θx_sindy = pool_data_test( x_noise, n_vars, poly_order ) 
            Ξ_sindy  = SINDy_test( x_noise, dx_noise, λ ) 
            Θx_gpsindy = pool_data_test(x_noise, n_vars, poly_order) 
            Ξ_gpsindy, hist_nvars = gpsindy( t, dx_noise, x_noise, λ, α, ρ, abstol, reltol )  
            # Ξ_gpsindy = Ξ_sindy ; hist_nvars = [] 

        # use noisy data  
        elseif case == 2 

            # add noise 
            println( "noise = ", noise ) 
            x_noise  = x_true + noise*randn( size(x_true, 1), size(x_true, 2) )
            dx_noise = dx_true + noise*randn( size(dx_true, 1), size(dx_true, 2) )

            Θx_sindy = pool_data_test( x_noise, n_vars, poly_order ) 
            Ξ_sindy  = SINDy_test( x_noise, dx_noise, λ ) 
            Θx_gpsindy = pool_data_test(x_noise, n_vars, poly_order) 
            Ξ_gpsindy, hist_nvars = gpsindy( t, dx_noise, x_noise, λ, α, ρ, abstol, reltol )  
            # Ξ_gpsindy = Ξ_sindy ; hist_nvars = [] 
    
        # use standardized true data 
        elseif case == 3 

            # set noise = standardized 
            x_true  = stand_data( t, x_true ) 
            dx_true = dx_true_fn( t, x_true, p, fn ) 
            x_noise = x_true ; dx_noise = dx_true  

            Θx_sindy = pool_data_test( x_noise, n_vars, poly_order ) 
            Ξ_sindy  = SINDy_test( x_noise, dx_noise, λ ) 
            Θx_gpsindy = pool_data_test(x_noise, n_vars, poly_order) 
            Ξ_gpsindy, hist_nvars = gpsindy( t, dx_noise, x_noise, λ, α, ρ, abstol, reltol )  
            # Ξ_gpsindy = Ξ_sindy ; hist_nvars = [] 

        # use standardized noisy data 
        elseif case == 4 

            # add noise 
            println( "noise = ", noise ) 
            x_true   = stand_data( t, x_true ) 
            dx_true  = dx_true_fn( t, x_true, p, fn ) 
            x_noise  = x_true + noise*randn( size(x_true, 1), size(x_true, 2) )
            dx_noise = dx_true + noise*randn( size(dx_true, 1), size(dx_true, 2) )

            Θx_sindy = pool_data_test( x_noise, n_vars, poly_order ) 
            Ξ_sindy  = SINDy_test( x_noise, dx_noise, λ ) 
            Θx_gpsindy = pool_data_test(x_noise, n_vars, poly_order) 
            Ξ_gpsindy, hist_nvars = gpsindy( t, dx_noise, x_noise, λ, α, ρ, abstol, reltol )  
            # Ξ_gpsindy = Ξ_sindy ; hist_nvars = [] 

        # standardize true x, finite difference dx 
        elseif case == 5 

            # set noise = standardized 
            x_true   = stand_data( t, x_true ) 
            dx_true  = dx_true_fn( t, x_true, p, fn ) 
            x_noise  = x_true 
            dx_noise = fdiff( t, x_true, 2 ) 

            Θx_sindy   = pool_data_test( x_noise, n_vars, poly_order ) 
            Ξ_sindy    = SINDy_test( x_noise, dx_noise, λ ) 
            Θx_gpsindy = pool_data_test(x_noise, n_vars, poly_order) 
            Ξ_gpsindy, hist_nvars = gpsindy( t, dx_noise, x_noise, λ, α, ρ, abstol, reltol )  
            # Ξ_gpsindy = Ξ_sindy ; hist_nvars = [] 

        # standardize and just use GP to smooth states (TEMPORAL) 
        elseif case == 6 
            
            # add noise 
            println( "noise = ", noise ) 
            x_true  = stand_data( t, x_true ) 
            dx_true = dx_true_fn( t, x_true, p, fn ) 
            x_noise  = x_true + noise*randn( size(x_true, 1), size(x_true, 2) )
            dx_noise = dx_true + noise*randn( size(dx_true, 1), size(dx_true, 2) )

            Θx_sindy = pool_data_test(x_noise, n_vars, poly_order) 
            Ξ_sindy  = SINDy_test( x_noise, dx_noise, λ ) 

            # smooth measurements 
            # t_test = collect( t[1] : 0.01 : t[end] )  
            t_test = t 
            x_GP,  Σ_test, hp_test = post_dist_SE( t, x_noise, t_test ) 
            dx_GP, Σ_test, hp_test = post_dist_SE( t, dx_noise, t_test ) 

            Θx_gpsindy = pool_data_test( x_GP, n_vars, poly_order ) 
            Ξ_gpsindy  = SINDy_test( x_GP, dx_GP, λ ) 

        # standardize --> smooth states into SINDy w/ GP (NON-temporal)  
        elseif case == 7 
            
            # add noise 
            println( "noise = ", noise ) 
            x_true   = stand_data( t, x_true ) 
            dx_true  = dx_true_fn( t, x_true, p, fn ) 
            x_noise  = x_true + noise*randn( size(x_true, 1), size(x_true, 2) )
            dx_noise = dx_true + noise*randn( size(dx_true, 1), size(dx_true, 2) )

            Θx_sindy = pool_data_test( x_noise, n_vars, poly_order ) 
            Ξ_sindy  = SINDy_test( x_noise, dx_noise, λ ) 

            # smooth measurements 
            x_GP, Σ_xsmooth, hp   = post_dist_SE( t, x_noise, t )  
            dx_GP, Σ_dxsmooth, hp = post_dist_SE( x_GP, dx_noise, x_GP )  
            
            Θx_gpsindy = pool_data_test(x_GP, n_vars, poly_order) 
            Ξ_gpsindy  = SINDy_test( x_GP, dx_GP, λ ) 

        # standardize --> smooth states w/ GP (NON-temporal) --> GPSINDy 
        elseif case == 8 
            
            # add noise 
            println( "noise = ", noise ) 
            x_true  = stand_data( t, x_true ) 
            dx_true = dx_true_fn( t, x_true, p, fn ) 
            x_noise  = x_true + noise*randn( size(x_true, 1), size(x_true, 2) )
            dx_noise = dx_true + noise*randn( size(dx_true, 1), size(dx_true, 2) )

            Θx_sindy = pool_data_test( x_noise, n_vars, poly_order ) 
            Ξ_sindy  = SINDy_test( x_noise, dx_noise, λ ) 

            # I guess .... let's try this again
            x_GP, Σ_xsmooth, hp = post_dist_SE( t, x_noise, t )  
            dx_GP, Σ_dxsmooth   = post_dist_SE( x_GP, dx_noise, x_GP )  
            
            Θx_gpsindy          = pool_data_test(x_GP, n_vars, poly_order) 
            Ξ_gpsindy, hist_nvars = gpsindy( t, dx_GP, x_GP, λ, α, ρ, abstol, reltol )  

        # same as 7, but GP --> SINDy --> GP --> SINDy 
        elseif case == 9 
            
            # add noise 
            println( "noise = ", noise ) 
            x_true   = stand_data( t, x_true ) 
            dx_true  = dx_true_fn( t, x_true, p, fn ) 
            x_noise  = x_true + noise*randn( size(x_true, 1), size(x_true, 2) )
            dx_noise = dx_true + noise*randn( size(dx_true, 1), size(dx_true, 2) )

            Θx_sindy = pool_data_test( x_noise, n_vars, poly_order ) 
            Ξ_sindy  = SINDy_test( x_noise, dx_noise, λ ) 
            
            # smooth measurements 
            x_GP, Σ_xGP, hp   = post_dist_SE( t, x_noise, t )           # step -1 
            dx_GP, Σ_dxGP, hp = post_dist_SE( x_GP, dx_noise, x_GP )    # step 0 
            
            Θx_gpsindy = pool_data_test(x_GP, n_vars, poly_order) 
            Ξ_gpsindy  = SINDy_test( x_GP, dx_GP, λ )                   # step 1 

            # step 2 
            dx_mean = Θx_gpsindy * Ξ_gpsindy 
            dx_post = dx_mean + (  ) * ( dx_noise - dx_mean )  

            Θx_gpsindy = pool_data_test(x_GP, n_vars, poly_order) 
            Ξ_gpsindy  = SINDy_test( x_GP, dx_post, λ ) 
                
        end 

        # plot 
        # plot_dx_sindy_gpsindy( t, dx_true, dx_noise, Θx_sindy, Ξ_sindy, Θx_gpsindy, Ξ_gpsindy ) 

        # metrics & diagnostics 
        Ξ_sindy_err, Ξ_gpsindy_err = l2_metric( n_vars, dx_noise, Θx_gpsindy, Ξ_true, Ξ_sindy, Ξ_gpsindy, sindy_err_vec, gpsindy_err_vec )
        push!( sindy_vec, Ξ_sindy ) 
        push!( gpsindy_vec, Ξ_gpsindy ) 


    end 

    # make matrices 
    Ξ_sindy_err   = mapreduce(permutedims, vcat, Ξ_sindy_err)
    Ξ_gpsindy_err = mapreduce(permutedims, vcat, Ξ_gpsindy_err)

    return Ξ_sindy_err, Ξ_gpsindy_err, hist_nvars_vec, Ξ_true, sindy_vec, gpsindy_vec 
end 
