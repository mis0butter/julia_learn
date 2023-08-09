# pendulum_test 

fn = double_pendulum
x0, str, p, ts, dt = init_params( fn )

plot_option = 1 
t, x = solve_ode( fn, x0, str, p, ts, dt, plot_option )

l1 = p[1] ; l2 = p[2] ; m1 = p[3] ; m2 = p[4] 

# get positions 
x1 = l1*sin.(x[:,1]);
y1 = -l1*cos.(x[:,1]);
x2 = l1*sin.(x[:,1])+l2*sin.(x[:,3]);
y2 = -l1*cos.(x[:,1])-l2*cos.(x[:,3]);

## ============================================ ##
# plot 

a = Animation() 

x_min = floor(minimum( [ minimum(x1), minimum(x2) ] )) 
x_max = ceil(maximum( [ maximum(x1), maximum(x2) ] )) 

y_min = floor(minimum( [ minimum(y1), minimum(y2) ] )) 
y_max = ceil(maximum( [ maximum(y1), maximum(y2) ] )) 

for i = 1 : size(x, 1)

    p = plot( 
        title = "Double Pendulum", 
        xlim  = ( x_min, x_max ), 
        ylim  = ( y_min, y_max ),  
     ) 

    scatter!( p, [0], [0], c = :gray, ms = 6, markerstrokewidth = 0 ) 
    plot!( p, [ 0, x1[i] ], [ 0, y1[i] ], c = :gray ) 
    plot!( p, [ x1[i], x2[i] ], [ y1[i], y2[i] ], c = :gray ) 

    plot!( p, x1[1:i], y1[1:i], c = :cyan, linealpha = 0.35 )
    scatter!( p, [x1[i]], [y1[i]], c = :cyan, ms = 6, markerstrokewidth = 0 ) 

    plot!( p, x2[1:i], y2[1:i], c = :orange, linealpha = 0.35 )
    scatter!( p, [x2[i]], [y2[i]], c = :orange, ms = 6, markerstrokewidth = 0 ) 
     
    frame(a, p)

end 

g = gif(a, fps = 10.0)
display(g)  



