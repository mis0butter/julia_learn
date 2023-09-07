## structs 

struct Hist 
    objval 
    fval 
    gval 
    hp 
    r_norm 
    s_norm 
    eps_pri 
    eps_dual 
end 

struct Ξ_struct 
    truth 
    sindy 
    gpsindy 
    gpsindy_x2 
end

struct Ξ_err_struct 
    sindy 
    gpsindy 
    gpsindy_x2 
end 

struct data_struct
    t 
    u 
    x_true 
    dx_true 
    x_noise 
    dx_noise 
end 

export Hist, Ξ_struct, Ξ_err_struct, data_struct 

