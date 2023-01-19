using SymbolicRegression

## init 
X = randn(Float32, 5, 100)
y = 2 * cos.(X[4, :]) + X[1, :] .^ 2 .- 2

## options 
options = SymbolicRegression.Options(
    binary_operators=[+, *, /, -],
    unary_operators=[cos, exp],
    npopulations=20
)

## hall of fame 
hall_of_fame = EquationSearch(
    X, y, niterations=40, options=options,
    parallelism=:multithreading
)

## check parento frontier 
dominating = calculate_pareto_frontier(X, y, hall_of_fame, options) 

## idk what this is 
trees = [member.tree for member in dominating]