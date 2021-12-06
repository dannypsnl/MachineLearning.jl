using Flux
using Flux: train!, @epochs
using Flux: Data.DataLoader

actual(x) = 4x + 2

x_train, x_test = hcat(0:5...), hcat(6:10...)
y_train, y_test = actual.(x_train), actual.(x_test)
data = DataLoader((x_train, y_train), shuffle=true)

model = Dense(1, 1)
loss(x, y) = Flux.Losses.mse(model(x), y)

opt = Descent()

@epochs 200 begin
  train!(loss, params(model), data, opt)
  @show(loss(x_test, y_test))
  @show(params(model))
end
