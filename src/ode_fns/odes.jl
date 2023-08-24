## ============================================ ##

export unicycle 
function unicycle( dx, x, p, t; u = [ 1/2*sin(t), cos(t) ] ) 
 
    v = x[3]    # forward velocity 
    θ = x[4]    # heading angle 

    dx[1] = v * cos(θ)      # x velocity 
    dx[2] = v * sin(θ)      # y velocity 
    dx[3] = u[1]            # acceleration  
    dx[4] = u[2]            # turn rate 
    
    return dx 
end 


## ============================================ ##

export dyn_car 
function dyn_car( xdot, xaug, p, t ) 
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
    # xdot = zeros(12, 1);
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
    xdot[7:12] .= 0 
    
    return xdot 
end 
    

## ============================================ ##

# Constants, I do like that I do not have to parse them manually to ode78
g = 9.81   # Acceleration due to gravity in m/s^2

export double_pendulum 
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

export ode_sine 
function ode_sine(dx, x, p, t)
    dx[1] = 1/2*sin(x[1])  
    # dx[2] = -1/2 * x[2] 
    return dx 
end 


## ============================================ ##

export pendulum 
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

export predator_prey 
function predator_prey(dx, (x1,x2), (a,b,c,d), t; u = 2sin(t) + 2sin(t/10))

    dx[1] = a*x1 - b*x1*x2 + u^2 * 0 
    dx[2] = -c*x2 + d*x1*x2  

    return dx 
end 


## ============================================ ##

export predator_prey_forcing 
function predator_prey_forcing(dx, (x1,x2), (a,b,c,d), t; u = 2sin(t) + 2sin(t/10))

    dx[1] = a*x1 - b*x1*x2 + u^2 
    dx[2] = -c*x2 + d*x1*x2  

    return dx 
end 


## ============================================ ##

export lorenz 
function lorenz(du, (x,y,z), (σ,ρ,β), t)

    du[1] = dx = σ * ( y - x ) 
    du[2] = dy = x * ( ρ - z ) - y 
    du[3] = dz = x * y - β * z  

    return du 
end 


## ============================================ ##

export quadcopter 
function quadcopter( dx, x, p, t; u = zeros(3) )
# ----------------------- #
# PURPOSE: 
#       quadcopter dynamics 
# INPUTS: 
#       dx 
#           dx[1:3]   = linear velocity 
#           dx[4:6]   = linear acceleration  
#           dx[7:9]   = angular velocity  
#           dx[10:12] = angular acceleration 
#       x 
#           x[1:3]    = linear position  
#           x[4:6]    = linear velocity   
#           x[7:9]    = angular rotation   
#           x[10:12]  = angular velocity  
#       p   = [ Ixx, Iyy, Izz, m ]
#       t   = time 
#       u   = control torque 
# OUTPUTS: 
#       dx 
#           dx[1:3]   = linear velocity 
#           dx[4:6]   = linear acceleration  
#           dx[7:9]   = angular velocity  
#           dx[10:12] = angular acceleration 
# ----------------------- #

    Ixx = p[1] 
    Iyy = p[2] 
    Izz = p[3] 
    m   = p[4]

    r  = x[1:3] 
    dr = x[4:6]
    ω  = x[7:9] 
    dω = x[10:12] 

    ddr = zeros(3) 
    R = rotation_euler( ω ) 

    # dummy variable for now 
    kd = -1 

    Fd = -kd * dr 
    ddr = [0, 0, -g] + ( 1 / m * u ) + Fd
    
    ddω = zeros(3) 
    ddω[1] = u[1] * Ixx^-1 - ( Iyy - Izz ) / ( Ixx ) * ω[2] * ω[3] 
    ddω[2] = u[2] * Iyy^-1 - ( Izz - Ixx ) / ( Iyy ) * ω[1] * ω[3]
    ddω[3] = u[3] * Izz^-1 - ( Ixx - Iyy ) / ( Izz ) * ω[1] * ω[2] 

    dx = [ dr, ddr, dω, ddω ] 

    return dx 
end 

export rotation_euler 
function rotation_euler( (ϕ, θ, ψ) )
# roll = ϕ, pitch = θ, yaw = ψ 

    cϕ = cos(ϕ)
    cθ = cos(θ)
    cψ = cos(ψ)
    sϕ = sin(ϕ)
    sθ = sin(θ)
    sψ = sin(ψ)

    R11 = cϕ * cψ - cθ * sϕ * sψ 
    R12 = -cψ * sϕ - cϕ * cθ * sψ 
    R13 = sθ * sψ 

    R21 = cθ * cψ * sϕ 
    R22 = cϕ * cθ * cψ - sϕ * sψ 
    R23 = -cψ * sθ 

    R31 = sϕ * sθ 
    R32 = cϕ * sθ 
    R33 = cθ 

    R = [ R11 R12 R13 ; 
          R21 R22 R23 ; 
          R31 R32 R33 ]
    
    return R 
end 



