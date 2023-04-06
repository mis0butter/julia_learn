module GP_june 

include("SINDy.jl")
include("GP_tools.jl")
include("lasso_admm.jl")


## ============================================ ##
# SINDy + GP objective function 

export f_obj 
function f_obj(( σ_f, l, σ_n, dx, ξ, Θx ))

    # training kernel function 
    Ky  = k_fn((σ_f, l, dx, dx)) + σ_n^2 * I 

    term  = 1/2*( dx - Θx*ξ )'*inv( Ky )*( dx - Θx*ξ ) 
    term += 1/2*log(det( Ky )) 

    return term 

end 


end 