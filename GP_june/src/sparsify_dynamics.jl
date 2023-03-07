export sparsify_dynamics 

function sparsify_dynamics( THETA, dx, lambda, n_vars ) 
    # ------------------------------------------------------------------------
    # Purpose: Solve for active terms in dynamics through sparse regression 
    # 
    # Inputs: 
    #   THETA  = data matrix 
    #   dx     = state derivatives 
    #   lambda = sparsification knob (threshold) 
    #   n_vars = # elements in state 
    # 
    # Outputs: 
    #   XI     = sparse coefficients of dynamics 
    # ------------------------------------------------------------------------
    
        # first perform least squares 
        XI = THETA \ dx ; 
    
        # sequentially thresholded least squares = LASSO. Do 10 iterations 
        for i = 1 : 10 
    
            # for each element in state 
            for j = 1 : n_vars 
    
                # small_inds = rows of XI < threshold 
                small_inds = findall( <(lambda), abs.(XI[:,j]) ) ; 
    
                println("small_inds = ", small_inds) 
                display(XI) 
    
                # set elements < lambda to 0 
                XI[small_inds, j] .= 0 ; 
    
                # big_inds --> select columns of THETA 
                # big_inds = ~small_inds ; 
    
            end 
    
        end 
        
    return XI 

end 
    