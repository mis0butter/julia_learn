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
    poly_order = n_vars 

    # construct data library 
    Θx = pool_data( [x u], n_vars, poly_order) 

    # first cut - SINDy 
    Ξ = sparsify_dynamics( Θx, dx, λ, n_vars-1 ) 

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
    m = size(xmat, 1) 

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

    # sine functions 
    for i = 1 : n_vars 
        ind  += 1 
        vec   = sin.(xmat[:,i]) 
        Θ = [Θ vec] 
    end 
    
    return Θ 

end 

## ============================================ ##
# build data matrix 

export pool_data2 

function pool_data2(xmat, n_vars, poly_order) 
# ----------------------- #
# Purpose: Build data matrix based on possible functions 
# 
# Inputs: 
#   x          = data input 
#   n_vars     = # elements in state 
#   poly_order = polynomial order (goes up to order 3) 
# 
# Outputs: 
#   Θ       = data matrix passed through function library 
# ----------------------- #

    # turn x into matrix and get length 
    # xmat = mapreduce(permutedims, vcat, x) 
    m = size(xmat, 1) 

    # fil out 1st column of Θ with ones (poly order = 0) 
    k = 1 ; 
    Θ = ones(m, k) 

    # poly order p 
    for p = 1 : poly_order 

        # current polynomial order 
        order  = factorial(n) / factorial( n-p ) 

        for i = 1 : order 

            k   += 1 
            vec  = 
            Θ    = [ Θ vec ]

        end 

    end 

    # sine functions 
    for i = 1 : n 
        k   += 1 
        vec  = sin.(xmat[:,i]) 
        Θ    = [Θ vec] 
    end 
    
    return Θ 

end 


## ============================================ ##

export n_vars_poly_order 
function n_vars_poly_order(x, k, Θ, n_vars, poly_order, p) 

    # loop through each state 
    for n = 1 : n_vars 

        println("p = ", p, ". k = ", k, ". n = ", n) 

        if p == 0 
            p = poly_order 
            println("resetting p = ", p)
            break 
        end 

        k  += 1 
        vec = x[:,n]  
        Θ   = [ Θ vec ]

        Θ   = n_vars_poly_order(x, k, Θ, n_vars, poly_order, p-1)
        
    end 

    return Θ 

end 

## ============================================ ##

export test_fn 
function test_fn( x, (a,b,c), y )

    y += 1 
    println("y = ", y)

    if y > 10 
        println("exit") 
        return x, a, c 
    end 

    x, a, c = test_fn(x, (a,b,c), y) 

    return x, a, c 
end 


## ============================================ ##

export recursion_fn 
function recursion_fn(x, Θ, k, i, v, poly_order) 

    n_vars = length(v) 

    # println("BEGIN fn. i = ", i, ". k = ", k, "\n" ) 

    # check if nest order is at poly order  
    if k < poly_order 

        # println("k < n_vars. i = ", i, ". k = ", k, "\n" )

        # increment nest order 
        k += 1 
        v[3] = k 
        for n = i : n_vars 

            # println("k += 1 --> ", k, ". n = ", n, "\n" )
            Θ  = recursion_fn(x, Θ, k, n, v, poly_order) 

        end 

    # nest order = poly order 
    else 

        # println("k = n_vars!!! i = ", i, ". k = ", k, "\n")

        # find all combinations of coupling according to poly_order 
        terms = factorial( poly_order + n_vars - 1 ) / 
        ( factorial(poly_order) * factorial(n_vars - 1) )

        for n = i : n_vars 
            println("n = ", n, ". i = ", i)
            vec = n 
            Θ = [ Θ vec ]
            println(Θ, "\n")
        end 

        return Θ
        
        # i += 1 
        # k  = 1 

    end 

    # println("END fn. i = ", i, ". k = ", k, "\n" ) 
  
    return Θ

end 


## ============================================ ##

export recursion_fn2 
function recursion_fn2(n) 

    println("n = ", n)

    if n == 1 
        return n 
    else 
        output = n * recursion_fn2(n-1) 
        println("n = ", n, ". n-1 = ", n-1, ". n*(n-1) = ", output)
        return output 
    end 

end 



## ============================================ ##

export recursion_fn3 
function recursion_fn3(x, Θ, v, poly_order) 

    # println("BEGIN. \nv = ", v, "\n")

    # end condition 
    if sum(v) > length(v).^2 

        return Θ

    else 

        # IF we have reached the last index for n_vars 
        if v[end] > poly_order 

            # println("reset v[end] = ", v[end])

            k = 1 
            # move back p IF index is at max n_vars 
            while v[end-k] == poly_order 
                k += 1 
            end 

            # increment higher level index, reset indices 
            v[end-k] += 1 
            v[end-k:end] .= v[end-k] 

            v = recursion_fn3(x, Θ, v, poly_order) 

            return v, Θ

        # loop through polynomials!!! 
        else 

            vec = 1 
            for i = 1:length(v) 
                vec = vec * x[(v[i])] 
            end 
            println("vec = ", vec) 
            Θ = [ Θ vec ]
            
            v[end] += 1 

            v = recursion_fn3(x, Θ, v, poly_order) 
            
            return v, Θ
        end 

    end 

end 
