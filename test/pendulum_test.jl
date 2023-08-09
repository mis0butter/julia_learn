# pendulum_test 

fn = double_pendulum
x0, str, p, ts, dt = init_params( fn )

plot_option = 1 
t, x = solve_ode( fn, x0, str, p, ts, dt, plot_option )

# plot 
x1 = l1*sin(x[:,1]);
y1 = -l1*cos(x[:,1]);
x2 = l1*sin(x[:,1])+l2*sin(x[:,3]);
y2 = -l1*cos(x[:,1])-l2*cos(x[:,3]);

