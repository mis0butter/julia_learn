## ============================================ ##
# putting it together (no control) 

export SINDy_test 
function SINDy_test( x, dx, λ, u = false )

    x_vars = size(x, 2)
    u_vars = size(u, 2) 
    poly_order = x_vars 
    
    if isequal(u, false)      # if u_data = false 
        n_vars = x_vars 
        data   = x 
    else            # there are u_data inputs 
        n_vars = x_vars + u_vars 
        data   = [ x u ]
    end 

    # construct data library 
    Θx = pool_data_test(data, n_vars, poly_order) 

    # first cut - SINDy 
    Ξ = sparsify_dynamics_test(Θx, dx, λ, x_vars) 

    return Ξ

end 


## ============================================ ##
# putting it together (with control) 

export SINDy_c_test 
function SINDy_c_test( x, u, dx, λ )

    x_vars = size(x, 2)
    u_vars = size(u, 2) 
    n_vars = x_vars + u_vars 
    poly_order = x_vars 

    # construct data library 
    Θx = pool_data_test( [x u], n_vars, poly_order ) 

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

    # poly order 2 
    if poly_order >= 2 
        for i = 1 : n_vars 
            for j = i : n_vars 
                ind += 1 ; 
                println( "i = ", i, ". j = ", j ) 
                vec  = xmat[:,i] .* xmat[:,j] 
                println("done") 
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

    # cos functions 
    for i = 1 : n_vars 
        ind  += 1 
        vec   = cos.(xmat[:,i]) 
        Θx    = [Θx vec] 
    end 

    # nonlinear combination with sine functions 
    for i = 1 : n_vars 
        for j = 1 : n_vars 
            ind  += 1 
            vec   = xmat[:,i] .* sin.(xmat[:,j]) 
            Θx    = [Θx vec]     
        end 
    end 

    # nonlinear combination with cosine functions 
    for i = 1 : n_vars 
        for j = 1 : n_vars 
            ind  += 1 
            vec   = xmat[:,i] .* cos.(xmat[:,j]) 
            Θx    = [Θx vec]     
        end 
    end 

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
    Θx = Vector{Function}(undef,0) 

    # fil out 1st column of Θ with ones (poly order = 0) 
    ind  = 1 
    push!(Θx, x -> 1) 

    # poly order 1 
    for i = 1 : n_vars 
        ind  += 1 
        push!( Θx, x -> x[i] ) 
    end 

    # ind += 1 
    # push!( Θ, x[1] .* x[2] )

    # poly order 2 
    if poly_order >= 2 
        for i = 1 : n_vars 
            for j = i:n_vars 
                ind += 1 ; 
                push!( Θx, x -> x[i] .* x[j] ) 
            end 
        end 
    end 

    # poly order 3 
    if poly_order >= 3 
        for i = 1 : n_vars 
            for j = i : n_vars 
                for k = j : n_vars 
                    ind += 1 ;                     
                    push!( Θx, x -> x[i] .* x[j] .* x[k] )
                end 
            end 
        end 
    end 

    # sine functions 
    for i = 1 : n_vars 
        ind  += 1
        push!(Θx, x -> sin.( x[i] ) )
    end 

    # sine functions 
    for i = 1 : n_vars 
        ind  += 1
        push!(Θx, x -> cos.( x[i] ) ) 
    end 

    # nonlinear combinations with sine functions 
    for i = 1 : n_vars 
        for j = 1 : n_vars 
            ind += 1 
            push!( Θx, x -> x[i] .* sin.( x[j] ) ) 
        end 
    end 

    # nonlinear combinations with cosine functions 
    for i = 1 : n_vars 
        for j = 1 : n_vars 
            ind += 1 
            push!( Θx, x -> x[i] .* cos.( x[j] ) ) 
        end 
    end 

    println("ind = ", ind)
    
    return Θx 

end 


## ============================================ ##
# export terms (with control)

export nonlinear_terms 
function nonlinear_terms( x_data, u_data = false ) 

    terms = [] 
    x_vars = size(x_data, 2) 
    u_vars = size(u_data, 2) 

    var_string = [] 
    for i = 1 : x_vars 
        push!( var_string, string("x", i) )
    end  
    if isequal(u_data, false)      # if u_data = false 
        n_vars = x_vars 
    else            # there are u_data inputs 
        n_vars = x_vars + u_vars 
        for i = 1 : u_vars 
            push!( var_string, string("u", i) ) 
        end 
    end 
    
    # first one 
    ind = 1  
    push!( terms, 1 )
    
     # poly order 1 
    for i = 1 : n_vars 
        ind += 1 
        push!( terms, var_string[i] ) 
    end 
    
    # poly order 2 
    for i = 1 : n_vars 
        for j = i : n_vars 
            ind += 1 
            push!( terms, string( var_string[i], var_string[j] ) ) 
        end 
    end 
    
     # poly order 3 
    for i = 1 : n_vars 
        for j = i : n_vars 
            for k = j : n_vars 
                ind += 1 
                push!( terms, string( var_string[i], var_string[j], var_string[k] ) )     
            end 
        end 
    end 
    
     # sine functions 
    for i = 1 : n_vars 
        ind += 1 
        push!( terms, string( "sin(", var_string[i], ")" ) ) 
    end 
    
     for i = 1 : n_vars 
        ind += 1 
        push!( terms, string( "cos(", var_string[i], ")" ) ) 
    end 
    
     # cosine functions 
    for i = 1 : n_vars 
        for j = 1 : n_vars 
            ind += 1 
            push!( terms, string( var_string[i], "sin(", var_string[j], ")") ) 
        end 
    end 
    
     for i = 1 : n_vars 
        for j = 1 : n_vars 
            ind += 1 
            push!( terms, string( var_string[i], "cos(", var_string[j], ")") ) 
        end 
    end 

    return terms 
end 


## ============================================ ##

export pretty_coeffs 
function pretty_coeffs(Ξ_true, x_true, u = false)

    # compute all nonlinear terms 
    terms = nonlinear_terms( x_true, u ) 

    # header 
    n_vars = size(x_true, 2) 
    header = [ "term" ] 
    for i = 1 : n_vars 
        push!( header, string( "x", i, "dot" ) ) 
    end 
    header = permutedims( header[:,:] ) 

    # unique inds of nonzero rows 
    n_vars = size(x_true, 2) 
    inds = [] 
    for i = 1 : n_vars 
        ind  = findall( x -> x > 0, Ξ_true[:,i] ) 
        inds = [ inds ; ind ]
    end 
    inds = sort(unique(inds)) 

    # save nzero rows 
    Ξ_nzero     = Ξ_true[inds, :]
    terms_nzero = terms[inds, :]
    
    # build Ξ with headers of nonzero rows 
    sz      = size(Ξ_nzero) 
    Ξ_terms = Array{Any}( undef, sz .+ (1,1) ) 
    Ξ_terms[1,:]          = header 
    Ξ_terms[2:end, 1]     = terms_nzero 
    Ξ_terms[2:end, 2:end] = Ξ_nzero   

    return Ξ_terms 

end 


