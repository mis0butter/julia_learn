fn = unicycle 

x0, dt, t, x_true, dx_true, dx_fd, p = ode_states(fn, 0, 2) 

λ = 0.1 
Ξ_true = SINDy_test( x_true, dx_true, λ ) 


