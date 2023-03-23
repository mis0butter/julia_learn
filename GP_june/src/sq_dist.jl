# define square distance function 
export sq_dist 

# define square distance function 
function sq_dist(a::Vector, b::Vector) 

    r = length(a) ; 
    p = length(b) 

    # iterate 
    C = zeros(r,p) 
    for i = 1:r 
        for j = 1:p 
            C[i,j] = ( a[i] - b[j] )^2 
        end 
    end 

    return C 

end 

