

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
            big_inds = findall( >(λ), abs.( Ξ[:,j] ) ) 

            # regress dynamics onto remaining terms to find sparse Ξ
            Ξ[big_inds, j] = Θx[:, big_inds] \ dx[:,j]

        end 

    end 
        
    return Ξ

end 

## ============================================ ##
# build data matrix 

export pool_data 

function pool_data(x, n_vars, poly_order) 
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
    xmat = mapreduce(permutedims, vcat, x) 
    m    = length(x) 

    # fil out 1st column of Θ with ones (poly order = 0) 
    ind = 1 ; 
    Θ = ones(m, ind) 

    # poly order 1 
    for i = 1 : n_vars 
        ind  += 1 
        Θ = [Θ xmat[:,i]]
    end 

    # poly order 2 
    if poly_order >= 2 
        for i = 1 : n_vars 
            for j = i:n_vars 

                ind  += 1 ; 
                vec   = xmat[:,i] .* xmat[:,j] 
                Θ = [Θ vec] 

            end 
        end 
    end 

    # poly order 3 
    if poly_order >= 3 
        for i = 1 : n_vars 
            for j = i : n_vars 
                for k = j : n_vars 
                    
                    ind  += 1 ;                     
                    vec   = xmat[:,i] .* xmat[:,j] .* xmat[:,k] 
                    Θ = [Θ vec] 

                end 
            end 
        end 
    end 

    # sine functions 
    for i = 1 : n_vars 
        ind  += 1 
        vec   = sin.(xmat[:,i]) 
        Θ = [Θ vec] 
    end 
    
    return Θ 

end 


