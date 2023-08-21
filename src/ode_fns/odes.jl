## ============================================ ##


function unicycle( dx, x, p, t; u = [ 1/2*sin(t), sin(t/10) ] ) 

    v = x[3] 
    θ = x[4] 

    dx[1] = v * cos(θ)
    dx[2] = v * sin(θ)
    dx[3] = u[1] 
    dx[4] = u[2] 
    
    return dx 
end 


## ============================================ ##


function dyn_car(t, xaug)
    #
    #Calculates the state derivative for a constant-velocity car modeled as a
    #box and driven at the back axle (back of the box).
    #
    #INPUTS:
    #   xaug - current vehicle state, augmented to include process noise.
    #   Consists of:
    #       x - vehicle x-position (back axle, m)
    #       y - vehicle y-position (back axle, m)
    #       s - vehicle forward speed (no-slip assumed)
    #       t - vehicle heading (rad.)
    #       l - vehicle length (m)
    #       w - vehicle width (m)
    #       ex - white noise driving x-position
    #       ey - white noise driving y-position
    #       es - white noise driving speed
    #       eh - white noise driving heading
    #       el - white noise driving vehicle length
    #       ew - white noise driving vehicle width
    #
    #OUTPUTS:
    #   xdot - state derivative
    
    ## calculate A 
    
    # x = sym('x', [12 1]); 
    # syms t 
    # dx = sym('dx', [12 1]); 
    # 
    # dx(1) = x(3) * cos(t) + x(7); 
    # dx(2) = x(3) * sin (t) + x(8); 
    # dx(3) = x(9); 
    # dx(4) = x(10); 
    # dx(5) = x(11); 
    # dx(6) = x(12); 
    # dx(7:12) = zeros(6,1); 
    
    ## state derivative 
    
    #extract speed from the state vector
    s = xaug[3]
    #extract heading from the state vector
    t = xaug[4]
    
    #extract the process noise from the augmented state vector
    ex = xaug[7] 
    ey = xaug[8]
    es = xaug[9]
    eh = xaug[10]
    el = xaug[11]
    ew = xaug[12]
    
    #calculate the state derivative
    xdot = zeros(12, 1);
    xdot[1] = s*cos(t) + ex 
    xdot[2] = s*sin(t) + ey 
    #assume speed is "constant," driven only by noise
    xdot[3] = es 
    #assume heading is "constant," driven only by noise
    xdot[4] = eh 
    #assume size is "constant," driven only by noise
    xdot[5] = el 
    xdot[6] = ew 
    #assume each of the process noise inputs are zero order hold
    xdot[7:12] = 0 
    
    return xdot 
end 
    

## ============================================ ##

# Constants, I do like that I do not have to parse them manually to ode78
g = 9.81   # Acceleration due to gravity in m/s^2


function double_pendulum( dx, x, p, t ) 

    l1 = p[1] 
    l2 = p[2] 
    m1 = p[3] 
    m2 = p[4] 

    θ₁  = x[1] 
    dθ₁ = x[2] 
    θ₂  = x[3] 
    dθ₂ = x[4] 

    # function [yprime] = pend(t, y)
    # yprime = zeros(4,1) 
    a = (m1 + m2) * l1 
    b = m2 * l2 * cos(θ₁ - θ₂) 
    c = m2 * l1 * cos(θ₁ - θ₂) 
    d = m2 * l2 
    e = -m2 * l2 * dθ₂ * dθ₂ * sin(θ₁ - θ₂) - g * (m1 + m2) * sin(θ₁) 
    f = m2 * l1 * dθ₁ * dθ₁ * sin(θ₁ - θ₂) - m2 * g * sin(θ₂) 
    
    dx[1] = dθ₁
    dx[2] = (e*d - b*f) / (a*d - c*b) 
    dx[3] = dθ₂ 
    dx[4] = (a*f - c*e) / (a*d - c*b) 
    # yprime = yprime' 
    # end

end 


## ============================================ ##


function ode_sine(dx, x, p, t)
    dx[1] = 1/2*sin(x[1])  
    # dx[2] = -1/2 * x[2] 
    return dx 
end 


## ============================================ ##


function pendulum(dx, x, p, t)

    θ  = x[1] 
    dθ = x[2] 

    l = p[1] 

	# The double pendulum equations
    # dx = [ 0.0; 0.0]
    dx[1] = dθ 
    dx[2] = -( g / l ) * cos(θ)

    # Return the derivatives as a vector
	return dx
end 


## ============================================ ##


function predator_prey(dx, (x1,x2), (a,b,c,d), t; u = 2sin(t) + 2sin(t/10))

    dx[1] = a*x1 - b*x1*x2 + u^2 * 0 
    dx[2] = -c*x2 + d*x1*x2  

    return dx 
end 


## ============================================ ##


function predator_prey_forcing(dx, (x1,x2), (a,b,c,d), t; u = 2sin(t) + 2sin(t/10))

    dx[1] = a*x1 - b*x1*x2 + u^2 
    dx[2] = -c*x2 + d*x1*x2  

    return dx 
end 


## ============================================ ##


function lorenz(du, (x,y,z), (σ,ρ,β), t)

    du[1] = dx = σ * ( y - x ) 
    du[2] = dy = x * ( ρ - z ) - y 
    du[3] = dz = x * y - β * z  

    return du 
end 



