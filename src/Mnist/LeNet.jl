using MLDatasets
using CUDA
using Flux
using Flux: Data.DataLoader
using Flux: onehotbatch, onecold, @epochs
using Statistics: mean

x_train, y_train = MNIST.traindata()
x_test, y_test = MNIST.testdata()
x_train = reshape(x_train, 28, 28, 1, :) .|> Float32
x_test = reshape(x_test, 28, 28, 1, :) .|> Float32
# one hot batch
y_train = onehotbatch(y_train, 0:9)
y_test = onehotbatch(y_test, 0:9)
x_train, x_test, y_train, y_test = if CUDA.functional()
    x_train |> gpu, x_test |> gpu, y_train |> gpu, y_test |> gpu
else
    x_train, x_test, y_train, y_test
end
data = DataLoader((x_train, y_train), batchsize=40, shuffle=true)

model = Chain(
  Conv((5, 5), 1=>6, relu),
  MaxPool((2, 2)),
  Conv((5, 5), 6=>16, relu),
  MaxPool((2, 2)),
  flatten,
  Dense(256, 120, relu), 
  Dense(120, 84, relu), 
  Dense(84, 10),
  softmax,
)
model = if CUDA.functional() model |> gpu else model end

loss(x, y) = Flux.crossentropy(model(x), y)
accurate(x, y) = mean(onecold(model(x)) .== onecold(y))
opt = ADAM()

@epochs 10 begin
  Flux.train!(loss, params(model), data, opt)
  @show(loss(x_test, y_test))
  @show(accurate(x_test, y_test))
end
