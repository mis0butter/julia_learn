fn = unicycle 

x0, dt, t, x_true, dx_true, dx_fd, p = ode_states(fn, 0, 2) 

λ = 0.1 
Ξ_true = SINDy_test( x_true, dx_true, λ ) 


## ============================================ ##
# pretty-ify coefficients table 

z = zeros(length(x_true), 2)
terms = nonlinear_terms( x_true, z ) 

## ============================================ ##

header = [ "term" ] 
for i = 1 : n_vars 
    push!( header, string( "x", i, "dot" ) ) 
end 
header = permutedims( header[:,:] ) 

sz = size(Ξ_true) 

Ξ_terms = Array{Any}( undef, sz .+ (1,1) )

Ξ_terms[1,:]          = header 
Ξ_terms[2:end, 1]     = terms 
Ξ_terms[2:end, 2:end] = Ξ_true  
 



