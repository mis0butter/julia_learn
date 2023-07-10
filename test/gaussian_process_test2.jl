using GaussianProcesses 
using Random 
using Optim 

Random.seed!(0) 

# create training data
n = 10 ;                        # number of training points 
x_train = 2π * rand(n) ;        # predictors 
y_train = 1 * sin.(x_train) + 0.1 * randn(n) ;  # measurements / data / regressors 

σ_f = 1.0
l   = 1.0 
σ_n = 0.1 

# mean and covariance 
mZero = MeanZero() ;            # zero mean function 
kern  = SE( log(σ_f) , log(l) ) ;          # squared eponential kernel (hyperparams on log scale) 
log_noise = log(σ_n) ;              # (optional) log std dev of obs noise 

# fit GP 
gp  = GP(x_train, y_train, mZero, kern, log_noise) ; 
plot(gp) 

test = optimize!(gp) 
plot(gp) 

