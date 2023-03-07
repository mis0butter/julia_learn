
export pool_data 

function pool_data(x, n_vars, poly_order) 
# ------------------------------------------------------------------------
# Purpose: Build data matrix based on possible functions 
# 
# Inputs: 
#   x           = data input 
#   n_vars      = # elements in state 
#   poly_order  = polynomial order (goes up to order 3) 
# 
# Outputs: 
#   THETA       = data matrix passed through function library 
# ------------------------------------------------------------------------

    # turn x into matrix and get length 
    xmat = mapreduce(permutedims, vcat, x) ; 
    m    = length(x) ; 

    # fil out 1st column of THETA with ones (poly order = 0) 
    ind = 1 ; 
    THETA = ones(m, ind) ; 

    # poly order 1 
    for i = 1 : n_vars 
        ind += 1 ; 
        THETA = [THETA xmat[:,i]]
    end 

    # poly order 2 
    if poly_order >= 2 
        for i = 1 : n_vars 
            for j = i:n_vars 

                ind += 1 ; 
                vec = xmat[:,i] .* xmat[:,j] ; 
                THETA = [THETA vec] ; 

            end 
        end 
    end 

    # poly order 3 
    if poly_order >= 3 
        for i = 1 : n_vars 
            for j = i : n_vars 
                for k = j : n_vars 
                    
                    ind += 1 ;                     
                    vec = xmat[:,i] .* xmat[:,j] .* xmat[:,k] ; 
                    THETA = [THETA vec] ; 

                end 
            end 
        end 
    end 

    # sine functions 
    for i = 1 : n_vars 
        ind += 1 ; 
        vec = sin.(xmat[:,i]) ; 
        THETA = [THETA vec] ; 
    end 
    
    return THETA 

end 

