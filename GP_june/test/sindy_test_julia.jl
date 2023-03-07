
## generate data 

x0 = [1; 1] ;           # initial conditions 
n_vars = size(x0, 1) ; 

dt = 0.01 ; 
ts = (0.0, 10.0) ; 
# ts = 0 : dt : 10.0 ;  

# define function 
function ode_test(dx, x, p, t)

    dx[1] = -1/4 * sin(x[1]) ; 
    dx[2] = -1/2 * x[2] ; 

    return dx 

end 

# DifferentialEquations.jl ODE problem 
using DifferentialEquations
ode = ODEProblem(ode_test, x0, ts) ; 
sol = solve(ode) ; 

t = sol.t ; 
x = sol.u ; 

## Compute Derivative 

# forwards difference 
dx_f = x * 0 ;  
for i in 1:lastindex(x) 
    if isequal(i, lastindex(x))    # if i == length(x) 
        dx_f[i] = ( x[i] - x[i-1] ) / ( t[i] - t[i-1] ) ; 
    else
        dx_f[i] = ( x[i+1] - x[i] ) / ( t[i+1] - t[i] ) ; 
    end 
end 

# central difference 
dx_c = x * 0 ; 
for i in 1:lastindex(x) 
    if isequal(i, 1) 
        dx_c[i] = ( x[i+1] - x[i] ) / ( t[i+1] - t[i] ) ; 
    elseif isequal(i, lastindex(x)) 
        dx_c[i] = ( x[i] - x[i-1] ) / ( t[i] - t[i-1] ) ; 
    else
        dx_c[i] = ( x[i+1] - x[i-1] ) / ( t[i+1] - t[i-1] ) ; 
    end 
end 

# backwards difference 
dx_b = x * 0 ;  
for i in 1:lastindex(x) 
    if isequal(i, 1)    # if i == 1 
        dx_b[i] = ( x[i+1] - x[i] ) / ( t[i+1] - t[i] ) ; 
    else
        dx_b[i] = ( x[i] - x[i-1] ) / ( t[i] - t[i-1] ) ; 
    end 
end 

# truth derivatives 
dx_t = x * 0 ; 
for i in 1:lastindex(x) 
    dx_t[i] = ode_test([0.0; 0.0], x[i], 0, 0) ; 
end 

println(dx_t) ; 

plot_option = 1 ; 
if isequal(plot_option, 1) 

end 

## Build library and compute sparse regression 

dx = dx_t ; 

# set parameters 
usesine = 1 ; 
xin = [x,t] ; 
polyorder = n_vars ; 

include("pool_data_julia.jl")


