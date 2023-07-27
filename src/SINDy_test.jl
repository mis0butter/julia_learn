## ============================================ ##
# putting it together (no control) 

export SINDy_test 
function SINDy_test( x, dx, λ )

    n_vars = size(x, 2) 
    poly_order = n_vars 

    # construct data library 
    Θx = pool_data_test(x, n_vars, poly_order) 

    # first cut - SINDy 
    Ξ = sparsify_dynamics_test(Θx, dx, λ, n_vars) 

    return Ξ

end 

## ============================================ ##
# putting it together (with control) 

export SINDy_c_test 
function SINDy_c_test( x, u, dx, λ )

    n_vars = size( [x u], 2 )
    x_vars = size(x, 2)
    u_vars = size(u, 2) 
    poly_order = n_vars 

    # construct data library 
    Θx = pool_data_test( [x u], n_vars, poly_order) 

    # first cut - SINDy 
    Ξ = sparsify_dynamics_test( Θx, dx, λ, x_vars ) 

    return Ξ

end 


## ============================================ ##
# solve sparse regression 

export sparsify_dynamics_test 
function sparsify_dynamics_test( Θx, dx, λ, n_vars ) 
# ----------------------- #
# Purpose: Solve for active terms in dynamics through sparse regression 
# 
# Inputs: 
#   Θx     = data matrix (of input states) 
#   dx     = state derivatives 
#   lambda = sparsification knob (threshold) 
#   n_vars = # elements in state 
# 
# Outputs: 
#   Ξ      = sparse coefficients of dynamics 
# ----------------------- #

    # first perform least squares 
    Ξ = Θx \ dx 

    # sequentially thresholded least squares = LASSO. Do 10 iterations 
    for k = 1 : 10 

        # for each element in state 
        for j = 1 : n_vars 

            # small_inds = rows of |Ξ| < λ
            small_inds = findall( <(λ), abs.(Ξ[:,j]) ) 

            # set elements < λ to 0 
            Ξ[small_inds, j] .= 0 

            # big_inds --> select columns of Θx
            big_inds = findall( >=(λ), abs.( Ξ[:,j] ) ) 

            # regress dynamics onto remaining terms to find sparse Ξ
            Ξ[big_inds, j] = Θx[:, big_inds] \ dx[:,j]

        end 

    end 
        
    return Ξ

end 

## ============================================ ##
# build data matrix 

export pool_data_test
function pool_data_test(xmat, n_vars, poly_order) 
# ----------------------- #
# Purpose: Build data matrix based on possible functions 
# 
# Inputs: 
#   x           = data input 
#   n_vars      = # elements in state 
#   poly_order  = polynomial order (goes up to order 3) 
# 
# Outputs: 
#   Θx          = data matrix passed through function library 
# ----------------------- #

    # turn x into matrix and get length 
    # xmat = mapreduce(permutedims, vcat, x) 
    l = size(xmat, 1) 

    # # fill out 1st column of Θx with ones (poly order = 0) 
    ind = 1 ; 
    Θx  = ones(l, ind) 

    # poly order 1 
    for i = 1 : n_vars 
        ind += 1 
        Θx   = [ Θx xmat[:,i] ]
    end 
    # Θx = xmat[:,1]
    # Θx = Θx[:,:] 
    # Θx = [ xmat[:,1] xmat[:,2] ]

    # ind += 1 
    # vec  = xmat[:,1] .* xmat[:,2]
    # Θx   = [ Θx vec[:,:] ]

    # poly order 2 
    if poly_order >= 2 
        for i = 1 : n_vars 
            for j = i:n_vars 

                ind += 1 ; 
                vec  = xmat[:,i] .* xmat[:,j] 
                Θx   = [Θx vec] 

            end 
        end 
    end 

    # poly order 3 
    if poly_order >= 3 
        for i = 1 : n_vars 
            for j = i : n_vars 
                for k = j : n_vars 
                    
                    ind += 1 ;                     
                    vec  = xmat[:,i] .* xmat[:,j] .* xmat[:,k] 
                    Θx   = [Θx vec] 

                end 
            end 
        end 
    end 

    # sine functions 
    for i = 1 : n_vars 
        ind  += 1 
        vec   = sin.(xmat[:,i]) 
        Θx    = [Θx vec] 
    end 

    println( "ind = ", ind ) 
    
    return Θx  

end 


## ============================================ ##
# build data matrix 

export pool_data_vecfn_test
function pool_data_vecfn_test(n_vars, poly_order) 
    # ----------------------- #
    # Purpose: Build data vector of functions  
    # 
    # Inputs: 
    #   n_vars      = # elements in state 
    #   poly_order  = polynomial order (goes up to order 3) 
    # 
    # Outputs: 
    #   Θ       = data matrix passed through function library 
    # ----------------------- #
    
    # initialize empty vector of functions 
    Θ = Vector{Function}(undef,0) 

    # fil out 1st column of Θ with ones (poly order = 0) 
    ind  = 1 
    push!(Θ, x -> 1) 

    # poly order 1 
    for i = 1 : n_vars 

        ind  += 1 
        push!( Θ, x -> x[i] ) 

    end 

    # ind += 1 
    # push!( Θ, x[1] .* x[2] )

    # poly order 2 
    if poly_order >= 2 
        for i = 1 : n_vars 
            for j = i:n_vars 

                ind += 1 ; 
                push!( Θ, x -> x[i] .* x[j] ) 

            end 
        end 
    end 

    # poly order 3 
    if poly_order >= 3 
        for i = 1 : n_vars 
            for j = i : n_vars 
                for k = j : n_vars 
                    
                    ind += 1 ;                     
                    push!( Θ, x -> x[i] .* x[j] .* x[k] )

                end 
            end 
        end 
    end 

    # sine functions 
    for i = 1 : n_vars 

        ind  += 1
        push!(Θ, x -> sin.( x[i] ) )

    end 
    
    return Θ 

end 
