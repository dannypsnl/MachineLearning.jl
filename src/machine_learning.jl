module machine_learning

using Flux
using Flux: train!, @epochs

actual(x) = 4x + 2

x_train, x_test = hcat(0:5...), hcat(6:10...)
y_train, y_test = actual.(x_train), actual.(x_test)

# stands for 1 input and 1 output
predict = Dense(1, 1)
loss(x, y) = Flux.Losses.mse(predict(x), y)

opt = Descent()
data = [(x_train, y_train)]
parameters = params(predict)

@epochs 200 begin
  train!(loss, parameters, data, opt)
  @show(loss(x_train, y_train))
  @show(parameters)
end

end # module
