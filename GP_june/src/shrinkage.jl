export shrinkage

# shrinkage 
function shrinkage(x, kappa) 

    z = 0*x ; 
    for i = 1:length(x) 
        z[i] = max( 0, x[i] - kappa ) - max( 0, -x[i] - kappa ) 
    end 

    return z 
end 