module GP_june 

struct Hist 
    objval 
    r_norm 
    s_norm 
    eps_pri 
    eps_dual 
end 

include("sq_dist.jl") 
include("gauss_sample.jl")
include("pool_data.jl")
include("sparsify_dynamics.jl")
include("lasso_admm.jl")

end 