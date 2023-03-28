
# sample from given mean and covariance 
export k_fn 

function k_fn(( σ_f, l, xp, xq ))

    return σ_f^2 * exp.( -1/( 2*l^2 ) * sq_dist(xp, xq) ) 

end 

