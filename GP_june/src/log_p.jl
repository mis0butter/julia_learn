
export log_p 

# marginal log-likelihood for Gaussian Process 
function log_p(( σ_f, l, σ_n, x, y, μ ))
    
    # kernel function 
    k_fn(σ_f, l, xp, xq) = σ_f^2 * exp.( -1/( 2*l^2 ) * sq_dist(xp, xq) ) 

    # training kernel function 
    Ky = k_fn(σ_f, l, x, x) 
    Ky += σ_n^2 * I 

    term = zeros(2)
    # term[1] = 1/2*( y )'*inv( Ky )*( y ) 
    term[1] = 1/2*( y .- μ )'*inv( Ky )*( y .- μ ) 
    term[2] = 1/2*log(det( Ky )) 

    return sum(term)

end 