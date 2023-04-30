
## ============================================ ##
# ODE functions 

export lorenz 
function lorenz(du, (x,y,z), (σ,ρ,β), t)

    du[1] = dx = σ * ( y - x ) 
    du[2] = dy = x * ( ρ - z ) - y 
    du[3] = dz = x * y - β * z  

    return du 
end 

export predator_prey 
function predator_prey(dx, (x1,x2), (a,b,c,d), t; u = 2sin(t) + 2sin(t/10))

    dx[1] = a*x1 - b*x1*x2 + u^2 * 0 
    dx[2] = -c*x2 + d*x1*x2  

    return dx 
end 

export ode_sine 
function ode_sine(dx, x, p, t)
    dx[1] = -1/4 * sin(x[1])  
    dx[2] = -1/2 * x[2] 
    return dx 
end 