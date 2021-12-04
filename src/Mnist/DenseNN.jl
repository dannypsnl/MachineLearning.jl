using MLDatasets
using Flux
using Flux: Data.DataLoader
using Flux: onehotbatch, onecold, throttle, @epochs
using Statistics: mean

x_train, y_train = MNIST.traindata()
x_test, y_test = MNIST.testdata()
# reshape image to 1D array
x_train = reshape(x_train, (28 * 28, :))
x_test = reshape(y_train, (28 * 28, :))
# one hot batch
y_train = onehotbatch(y_train, 0:9)
y_test = onehotbatch(y_test, 0:9)
data = DataLoader((x_train, y_train), batchsize=40, shuffle=true)

model = Chain(
  Dense(28 * 28, 40, relu),
  Dense(40, 10),
  softmax
)

loss(x, y) = Flux.crossentropy(model(x), y)
acc(x, y) = mean(Flux.onecold(model(x)) .== Flux.onecold(y))
opt = ADAM()

@epochs 10 begin
  Flux.train!(loss, params(model), data, opt)
  @show(loss(x_test, y_test))
  @show(acc(x_test, y_test))
end
