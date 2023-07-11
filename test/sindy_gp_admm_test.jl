using GaussianSINDy

## ============================================ ##

λ = 0.1 
abstol = 1e-2 ; reltol = 1e-2           
noise_vec = 0 
sindy_err_vec, gpsindy_err_vec, hist_nvars_vec = monte_carlo_gpsindy( noise_vec, λ, abstol, reltol ) 



