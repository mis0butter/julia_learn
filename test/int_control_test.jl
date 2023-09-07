
# function unicycle( dx, x, p, t; u = [ 1/2*sin(t), cos(t) ] ) 
 
#     v = x[3]    # forward velocity 
#     θ = x[4]    # heading angle 

#     dx[1] = v * cos(θ)      # x velocity 
#     dx[2] = v * sin(θ)      # y velocity 
#     dx[3] = u[1]            # acceleration  
#     dx[4] = u[2]            # turn rate 
    
#     return dx 
# end 

# ----------------------- #
# function dx_true_fn(t, x, p, fn)

#     # true derivatives 
#     dx_true = 0*x
#     n_vars  = size(x, 2) 
#     z       = zeros(n_vars) 

#     for i = 1 : length(t) 
#         dx_true[i,:] = fn( z, x[i,:], p, t[i] ) 
#     end 

#     return dx_true 


## ============================================ ##

using GaussianSINDy 

fn = unicycle 

x0, dt, t, x_true, dx_true, dx_fd, p = ode_states(fn, 0, 2) 

N = length(t) 
u = rand(N, 2) 
z = zeros(size(x0,1)) 

xt = x0 
x_hist = [] 
push!( x_hist, xt )
for i = 1 : N 

    ut = u[i,:]
    xt == xt + dt * fn( z, xt, p, t[i] ) 
    push!( x_hist, xt )

end 


