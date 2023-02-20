using GP_june 

# true hyperparameters 
sig_f0 = 1 ; 
l0     = 1 ; 
sig_n0 = 0.1 ; 

# generate training data 
N = 5 ; 
x_train = sort( 10*rand(N,1), dims=1 )      # matrix     
x_train = [ 1.0343 2.8932 4.1403 5.1443 5.3743 ] ;  # SAME AS MATLAB 


