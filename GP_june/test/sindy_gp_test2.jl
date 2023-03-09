using Revise 
using DifferentialEquations 
using GP_june 
using LinearAlgebra 
using Zygote 
using Optim 


## ============================================ ##
# loss function 

x = rand(3) 
A = rand(3,3) 
xAx = x'*A*x 

loss( σ )  = σ^2 * x'*A*x
loss2( σ ) = σ[1]^2 * xAx + σ[2]

σ0 = [1.0, 1.0] 
result = optimize( loss2, σ0 )
println("result = ", result) 

## ============================================ ##

f(x) = (1.0 - x[1])^2 + 100.0 * (x[2])^2
x0 = [0.0, 0.0]
result = optimize(f, x0)
println(result) 

## ============================================ ##

dx = rand(3) 
Θ  = rand(3,3)
sig_f = 1.0  
loss( ξ ) = 1/2*( dx - Θ*ξ )'*inv( sig_f^2 )*( dx - Θ*ξ  ) 

ξ0 = rand(3) 
result = optimize(loss, ξ0) 
println("minimizer = ", result.minimizer) 

## ============================================ ##

dx = rand(3) 
Θ  = rand(3,3)
ξ  = rand(3) 
loss( σ ) = 1/2*( dx - Θ*ξ )'*inv( σ[1]^2*σ[2] )*( dx - Θ*ξ  ) 

σ0 = rand(2) 
result = optimize(loss, σ0) 
println("minimizer = ", result.minimizer) 

## ============================================ ##

dx = rand(3) 
Θ  = rand(3,3)
ξ  = rand(3) 
loss(( sig_f, l )) = 1/2*( dx - Θ*ξ )'*inv( sig_f^2*l )*( dx - Θ*ξ  ) 

σ0 = rand(2) 
result = optimize(loss, σ0) 
println("minimizer = ", result.minimizer) 

## ============================================ ##

dx = rand(3) 
Θ  = rand(3,3)
ξ  = rand(3) 
loss(( sig_f, l )) = 1/2*( dx - Θ*ξ )'*inv( sig_f^2*l )*( dx - Θ*ξ  ) 

σ0 = rand(2) 
result = optimize(loss, σ0) 
println("minimizer = ", result.minimizer) 

## ============================================ ##

dx = rand(3) 
Θ  = rand(3,3)
ξ  = rand(3) 
loss(( sig_f, l )) = 1/2*( dx - Θ*ξ )'*inv( sig_f^2 * exp(-1/(2*l^2)) )*( dx - Θ*ξ  ) 

σ0 = rand(2) 
result = optimize(loss, σ0) 
println("minimizer = ", result.minimizer) 

## ============================================ ##

n  = 10
dx = rand(n) 
Θ  = rand(n,n)
ξ  = rand(n) 
log_Z(( sig_f, l )) = 1/2*( dx - Θ*ξ )'*inv( sig_f^2 * exp( -1/(2*l^2) * sq_dist(dx,dx) ) )*( dx - Θ*ξ  ) + 1/2*log(det( sig_f^2 * exp( -1/(2*l^2) * sq_dist(dx,dx) ) ))

σ0 = [1.0, 1.0]
result = optimize(log_Z, σ0) 
println("minimizer = ", result.minimizer) 

## ============================================ ##

n  = 10
dx = rand(n) 
Θ  = rand(n,n)
ξ  = rand(n) 
log_Z(( sig_f, l, sig_n )) = 1/2*( dx - Θ*ξ )'*inv( sig_f^2 * exp( -1/(2*l^2) * sq_dist(dx,dx) ) + sig_n^2 * I )*( dx - Θ*ξ  ) + 1/2*log(det( sig_f^2 * exp( -1/(2*l^2) * sq_dist(dx,dx) ) ))

σ0 = [1.0, 1.0, 1.0]
result = optimize(log_Z, σ0) 
println("minimizer = ", result.minimizer) 



