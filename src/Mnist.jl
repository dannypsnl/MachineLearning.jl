module Mnist

using MLDatasets
using Flux
using Flux: onehotbatch
using Flux: crossentropy, onecold, throttle
using Flux: @epochs

x_train, y_train = MNIST.traindata()
x_test,  y_test  = MNIST.testdata()

model = Chain(
  Dense(28 * 28, 40, relu),
  Dense(40, 10),
  softmax
)

loss(X, y) = crossentropy(model(X), y)
progress = () -> @show(loss(X, y))
opt = ADAM()
data = [(x_train, y_train)]

@epochs 100 Flux.train!(loss, params(model), data, opt, cb = throttle(progress, 10))

end