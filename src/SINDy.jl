## ============================================ ##
# putting it together (no control) 

export SINDy 
function SINDy( x, dx, λ )

    n_vars = size(x, 2) 
    poly_order = n_vars 

    # construct data library 
    Θx = pool_data(x, n_vars, poly_order) 

    # first cut - SINDy 
    Ξ = sparsify_dynamics(Θx, dx, λ, n_vars) 

    return Ξ

end 

## ============================================ ##
# putting it together (with control) 

export SINDy_c 
function SINDy_c( x, u, dx, λ )

    n_vars = size( [x u], 2 )
    u_vars = size(u, 2) 
    poly_order = n_vars 

    # construct data library 
    Θx = pool_data( [x u], n_vars, poly_order) 

    # first cut - SINDy 
    Ξ = sparsify_dynamics( Θx, dx, λ, n_vars-u_vars ) 

    return Ξ

end 


## ============================================ ##
# solve sparse regression 

export sparsify_dynamics 
function sparsify_dynamics( Θx, dx, λ, n_vars ) 
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

export pool_data 

function pool_data(xmat, n_vars, poly_order) 
# ----------------------- #
# Purpose: Build data matrix based on possible functions 
# 
# Inputs: 
#   x           = data input 
#   n_vars      = # elements in state 
#   poly_order  = polynomial order (goes up to order 3) 
# 
# Outputs: 
#   Θ       = data matrix passed through function library 
# ----------------------- #

    # turn x into matrix and get length 
    # xmat = mapreduce(permutedims, vcat, x) 
    l = size(xmat, 1) 

    # fil out 1st column of Θ with ones (poly order = 0) 
    ind = 1 ; 
    Θ = ones(l, ind) 

    # poly order 1 
    for i = 1 : n_vars 
        ind  += 1 
        Θ = [Θ xmat[:,i]]
    end 

    # poly order 2 
    if poly_order >= 2 
        for i = 1 : n_vars 
            for j = i:n_vars 

                ind += 1 ; 
                vec  = xmat[:,i] .* xmat[:,j] 
                Θ    = [Θ vec] 

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
                    Θ    = [Θ vec] 

                end 
            end 
        end 
    end 

    # # sine functions 
    # for i = 1 : n_vars 
    #     ind  += 1 
    #     vec   = sin.(xmat[:,i]) 
    #     Θ = [Θ vec] 
    # end 
    
    return Θ 

end 


## ============================================ ##
# build data matrix 

export pool_data_vecfn
function pool_data_vecfn(n_vars, poly_order) 
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

    # # sine functions 
    # for i = 1 : n_vars 

    #     ind  += 1
    #     push!(Θ, x -> sin.( x[i] ) )

    # end 
    
    return Θ 

end 


## ============================================ ##

export pool_data_rwrap
function pool_data_rwrap( x_in, n_vars, poly_order )

    l      = size(x_in, 1) 
    # n_vars = size(x_in, 2) 

    # construct data library 
    Θx = ones(l, 1)
    for p = 1 : poly_order 
        Θx_p, = pool_data_recursion( x_in, p) 
        Θx    = [ Θx Θx_p ]
    end 

    # # append sine functions 
    # for i = 1 : n_vars 
    #     vec = sin.(x_in[:,i]) 
    #     Θx  = [Θx vec] 
    # end 

    return Θx 
end 


## ============================================ ##
# putting it together (with control) 

export SINDy_c_recursion
function SINDy_c_recursion(x, dx, u, λ, poly_order )

    if u == 0 
        n_vars = size(x, 2) 
        u_vars = 0 
        x_in   = x 
    else 
        n_vars = size( [x u], 2 )
        u_vars = size(u, 2) 
        x_in   = [ x u ]
    end 
    x_vars = n_vars - u_vars 
    h = [] 
    for i = 1:x_vars 
        push!(h, "dx$(i)")
    end 
    
    # construct data library 
    Θx = pool_data_rwrap( x_in, n_vars, poly_order )

    # SINDy 
    Ξ = sparsify_dynamics( Θx, dx, λ, x_vars ) 

    return Ξ

end 


## ============================================ ##

export pool_data_recursion  
function pool_data_recursion( x, poly_order, Θ = Array{Float64}(undef, size(x,1), 0), v_ind = Int.(ones(poly_order)) ) 
# ----------------------- #
# Purpose: Build data matrix based on possible functions 
# 
# Inputs: 
#   x          = data input 
#   poly_order = polynomial order 
#   Θ          = data matrix passed through function library (optional) 
#   v_ind      = vector of indices (optional)  
# 
# Outputs: 
#   Θ          = data matrix passed through function library 
#   v_ind      = vector of indices (optional)  
# ----------------------- #

    # set-up for end condition checking 
    n_vars = size(x,2) 
    terms  = factorial( poly_order + n_vars - 1 ) / 
             ( factorial(poly_order) * factorial(n_vars - 1) )
    Θ_n    = size(Θ, 2)
    
    # end condition 
    if Θ_n == terms 

        return Θ, v_ind 

    # recursion 
    else 

        # IF we have reached the last index for n_vars 
        if v_ind[end] > n_vars 

            # move back k IF index is at max n_vars 
            k = 1 
            while v_ind[end-k] == n_vars  
                k += 1 
            end 

            # increment higher level index, reset indices 
            v_ind[end-k]     += 1 
            v_ind[end-k:end] .= v_ind[end-k] 

            # back into the rabbit hole 
            Θ, v_ind = pool_data_recursion(x, poly_order, Θ, v_ind) 

            return Θ, v_ind

        # loop through polynomials!!! 
        else 

            # couple state variables!!! 
            vec = ones(size(x,1),1) 
            for i = 1 : poly_order 
                vec = vec .* x[:, v_ind[i] ] 
            end 

            # add to data matrix 
            Θ = [ Θ vec[:,:] ] 
            # println("Θ = ") ; display(Θ)
            
            # increment last index 
            v_ind[end] += 1 

            # continue recursion 
            Θ, v_ind = pool_data_recursion(x, poly_order, Θ, v_ind) 
            
            return Θ, v_ind
        end 

    end 

end 

## ============================================ ##

export fibonacci 
function fibonacci(n)

    # base condition 
    if n <= 1 
        return 0 
    elseif n == 2 
        return 1 
    else 
        out = fibonacci(n-1) + fibonacci(n-2) 
        return out 
    end 

end 
    
    

