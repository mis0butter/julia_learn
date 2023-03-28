export post_dist

function post_dist(( x_train, y_train, x_test, σ_f, l, σ_n ))

    # x  = training data  
    # xs = test data 
    # joint distribution 
    #   [ y  ]     (    [ K(x,x)+Ïƒ_n^2*I  K(x,xs)  ] ) 
    #   [ fs ] ~ N ( 0, [ K(xs,x)         K(xs,xs) ] ) 

    # covariance from training data 
    K    = k_fn(σ_f, l, x_train, x_train)  
    K   += σ_n^2 * I                        # add noise for positive definite 
    Ks   = k_fn(σ_f, l, x_train, x_test)  
    Kss  = k_fn(σ_f, l, x_test, x_test) 

    # conditional distribution 
    # mu_cond    = K(Xs,X)*inv(K(X,X))*y
    # sigma_cond = K(Xs,Xs) - K(Xs,X)*inv(K(X,X))*K(X,Xs) 
    # fs | (Xs, X, y) ~ N ( mu_cond, sigma_cond ) 
    μ_post = Ks' * K^-1 * y_train 
    Σ_post = Kss - (Ks' * K^-1 * Ks)  

    return μ_post, Σ_post

end 