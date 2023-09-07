using Flux

export train_nn_predict
function train_nn_predict(x_train, y_train, n_epochs, n_in_features)

    model = Chain(Dense(n_in_features => 32, relu), Dense(32 => 1, bias=false), only)
    model = f64(model)

    loss(x, y) = Flux.mse(model(x), y)

    # Training data
    data = [(x_train[i, :], y_train[i, 1]) for i in 1:size(x_train, 1)]

    # Training loop
    optim = Flux.setup(Adam(), model)
    for epoch in 1:n_epochs
        Flux.train!((m, x, y) -> (m(x) - y)^2, model, data, optim)
        # Print the avergae loss for the epoch
        avg_loss = sum([loss(x, y) for (x, y) in data]) / length(data)
        println("Epoch $epoch. Average Loss: $avg_loss")
    end

    # Make predictions
    y_pred = [model(x_train[i, :]) for i in 1:size(x_train, 1)]
    return y_pred
end