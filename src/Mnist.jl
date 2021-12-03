module Mnist

using MLDatasets
using Flux, CUDA
using Flux: Data.DataLoader
using Flux: onehotbatch, onecold, throttle, @epochs
using Statistics: mean

x_train, y_train = MNIST.traindata()
x_test, y_test = MNIST.testdata()
# reshape to 1D array
x_train = reshape(x_train, (28 * 28, :)) |> gpu
x_test = reshape(y_train, (28 * 28, :)) |> gpu
# one hot batch
y_train = onehotbatch(y_train, 0:9) |> gpu
y_test = onehotbatch(y_test, 0:9) |> gpu
data = DataLoader((x_train, y_train), batchsize=40, shuffle=true)

model = Chain(
  Dense(28 * 28, 40, relu),
  Dense(40, 10),
  softmax
) |> gpu

loss(x, y) = Flux.crossentropy(model(x), y)
acc(x, y) = mean(Flux.onecold(model(x)) .== Flux.onecold(y))
function progress()
  @show(loss(x_test, y_test))
  @show(acc(x_test, y_test))
end
opt = ADAM()

@epochs 10 Flux.train!(loss, params(model), data, opt, cb = throttle(progress, 2))

end
