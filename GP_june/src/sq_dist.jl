# define square distance function 
export sq_dist 
function sq_dist(a::Matrix, b::Matrix) 

    # extract dims 
    D, n = size(a) ;  
    d, m = size(b) ;

    # ensure a, b are "row" matrices 
    if D > n 
        a = transpose(a) ; 
        D, n = size(a) ; 
    end 
    if d > m 
        b = transpose(b) ; 
        d, m = size(b) ; 
    end 

    # iterate 
    C = zeros(n,m) ; 
    for d = 1:D 
        amat = repeat(transpose(a), outer = [1,m]) ; 
        bmat = repeat(b, outer = [n,1]) ; 
        C += (bmat - amat).^2 ; 
    end 

    return C 

end 

