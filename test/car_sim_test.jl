fn = unicycle 

x0, dt, t, x_true, dx_true, dx_fd, p = ode_states(fn, 0, 2) 

λ = 0.1 
Ξ_true = SINDy_test( x_true, dx_true, λ ) 


## ============================================ ##
# pretty-ify coefficients table 

terms = [] 
n_vars = size(x_true, 2) 

# first one 
ind = 1  
push!( terms, 1 )

 # poly order 1 
for i = 1 : n_vars 
    ind += 1 
    push!( terms, string( "x", i )  ) 
end 

# poly order 2 
for i = 1 : n_vars 
    for j = i : n_vars 
        ind += 1 
        push!( terms, string( "x", i, "x", j ) ) 
    end 
end 

 # poly order 3 
for i = 1 : n_vars 
    for j = i : n_vars 
        for k = j : n_vars 
            ind += 1 
            push!( terms, string( "x", i, "x", j, "x", k ) )     
        end 
    end 
end 

 # sine functions 
for i = 1 : n_vars 
    ind += 1 
    push!( terms, string( "sin(x", i, ")" ) ) 
end 

 for i = 1 : n_vars 
    ind += 1 
    push!( terms, string( "cos(x", i, ")" ) ) 
end 

 # cosine functions 
for i = 1 : n_vars 
    for j = 1 : n_vars 
        ind += 1 
        push!( terms, string( "x", i, "sin(x", j, ")") ) 
    end 
end 

 for i = 1 : n_vars 
    for j = 1 : n_vars 
        ind += 1 
        push!( terms, string( "x", i, "cos(x", j, ")") ) 
    end 
end 

## ============================================ ##

header = [ "term" ] 
for i = 1 : n_vars 
    push!( header, string( "x", i, "dot" ) ) 
end 

sz = size(Ξ_true) 

Ξ_terms = Array{Any}( undef, sz .+ (0,1) )

Ξ_terms[:, 1] = terms 
Ξ_terms[:, 2:end] = Ξ_true  

pretty_table( Ξ_terms; header = header ) 
 



