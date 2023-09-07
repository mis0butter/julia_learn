using DifferentialEquations
using LinearAlgebra: dot

# returns weights
ξ() = [0.1, 0.2, 0.3, 0.4]

# define functions
f1(x) = 1
f2(x) = x^2
f3(x) = cos(x)
f4(x) = sin(x)

# create a vector of functions (can be done dynamically)
fn_vector = [ f1, f2, f3, f4 ]

# numerically evaluate each function at x and return a vector of numbers
𝚽(x, fn_vector) = [ f(x) for f in fn_vector ]

# function solve_ode()

## ============================================ ##

# define the differential equation
f(x,p,t) = dot(𝚽(x, fn_vector), ξ())

# setup the problem
x0 = 1.0
tspan = (0.0, 1.0)
prob = ODEProblem(f, x0, tspan)

# solve the ODE
sol = solve(prob,  reltol = 1e-8, abstol = 1e-8)

# print the solution
println("Solution at t = 1.0 is: ", sol(1.0))

# end