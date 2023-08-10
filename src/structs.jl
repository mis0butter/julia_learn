# =====================================================================
# === Dynamics Basis

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

struct Ξ_hist
    truth 
    sindy 
    gpsindy 
    gpsindy_x2 
end

export Hist, Ξ_hist 

