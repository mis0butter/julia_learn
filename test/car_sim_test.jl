fn = unicycle 

x0, dt, t, x_true, dx_true, dx_fd, p = ode_states(fn, 0, 2) 

λ = 0.1 
Ξ_true = SINDy_test( x_true, dx_true, λ ) 


## ============================================ ##
# pretty-ify coefficients table 

terms = [] 

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
    for j = 1 : n_vars 
        ind += 1 
        push!( terms, string( "x", i, "x", j ) ) 
    end 
end 

# poly order 3 
for i = 1 : n_vars 
    for j = 1 : n_vars 
        for k = 1 : n_vars 
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
        push!( terms, string( "x", i, "sin(x", j, ")") ) 
    end 
end 

for i = 1 : n_vars 
    for j = 1 : n_vars 
        push!( terms, string( "x", i, "cos(x", j, ")") ) 
    end 
end 





