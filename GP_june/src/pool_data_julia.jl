export pool_data_julia 
function pool_data_julia(X, n_vars, poly_order)
# ------------------------------------------------------------------------
# Purpose: Build data matrix based on possible functions() 
# 
# Inputs: 
#   X           = data input() 
#   n_vars      = # elements in state 
#   poly_order  = polynomial order [goes up to order 3] 
# 
# Outputs: 
#   THETA       = data matrix passed through function library() 
#   terms       = terms in function library (strings) 
#   states      = elements in state [strings] 
# ------------------------------------------------------------------------
    
# Save vars to build data matrix 
m = length(X); 

# build output states string
states = Any[] ; 
for i in 1 : n_vars 
    temp = string( "x", string(i) ) ; 
    push!( states, temp )
end 

# fill out 1st column of THETA with ones (poly_order = 0) 
ind = 1; 
THETA = Any[] 
THETA[:, ind] = ones(m, ind); 
terms = Any[] ; 
terms(ind, 1) = '1'; 

# poly_order = 1 
for i in 1 : n_vars 
    
    # increment ind [column of data matrix] 
    ind = ind + 1; 
    
    # poly_order = 1
    THETA[:, ind] = X[:,i]; 
    terms(ind, 1) = ['x' num2str(i)]; 
    
end 

# poly_order = 2 
if poly_order >= 2 
    for i = 1 : n_vars 
        for j = i : n_vars 

        # increment ind [column of data matrix] 
        ind = ind + 1; 

        # poly_order = 2 
        THETA[:, ind] = X[:,i] .* X[:,j]; 
        terms(ind, 1) = ["x' num2str(i) 'x" num2str(j)]; 

        end 
    end 
end 

# poly_order = 3 
if poly_order >= 3 
    for i = 1 : n_vars 
        for j = i : n_vars 
            for k = j : n_vars 
                
                # increment ind [column of data matrix] 
                ind = ind + 1; 
                
                # poly_order = 3 
                THETA[:, ind] = X[:,i] .* X[:,j] .* X[:,k]; 
                terms(ind, 1) = ["x' num2str(i) 'x' num2str(j) 'x" num2str(k)]; 
    
            end 
        end 
    end 
end 

# sine fns 
for i = 1 : n_vars 
    
    # increment ind 
    ind = ind + 1; 
    
    # check sine 
    THETA[:, ind] = sin(X[:,i]); 
    terms(ind, 1) = ["sin(x' num2str(i) ')"]; 
    
end 

return THETA, terms, states 

end 