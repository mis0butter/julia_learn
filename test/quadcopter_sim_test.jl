fn = quadcopter 

x0, dt, t, x_true, dx_true, dx_fd, p = ode_states(fn, 0, 2) 

## ============================================ ##


u = [] 
for i = 1 : length(t) 
    push!( u, [ 1/2*sin(t[i]), sin(t[i]/10) ] ) 
end 
u = vv2m(u) 

λ = 0.1 
Ξ_true  = SINDy_test( x_true, dx_true, λ, u ) 
Ξ_true_terms = pretty_coeffs(Ξ_true, x_true, u) 

 
## ============================================ ##


fn = unicycle 
x0, dt, t, x_true, dx_true, dx_fd, p = ode_states(fn, 0, 2) 

u = [] 
for i = 1 : length(t) 
    push!( u, [ 1/2*sin(t[i]), sin(t[i]/10) ] ) 
end 
u = vv2m(u) 

noise = 0.01 
x_noise  = x_true + noise*randn( size(x_true, 1), size(x_true, 2) )
dx_noise = dx_true + noise*randn( size(dx_true, 1), size(dx_true, 2) )

λ = 0.1 
Ξ_noise = SINDy_test( x_noise, dx_noise, λ, u ) 
Ξ_noise_terms = pretty_coeffs(Ξ_noise, x_true, u) 



