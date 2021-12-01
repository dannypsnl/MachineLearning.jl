module machine_learning

using Flux
using Flux: train!

actual(x) = 4x + 2

x_train, x_test = hcat(0:5...), hcat(6:10...)
y_train, y_test = actual.(x_train), actual.(x_test)

# stands for 1 input and 1 output
predict = Dense(1, 1)
loss(x, y) = Flux.Losses.mse(predict(x), y)

opt = Descent()
data = [(x_train, y_train)]
parameters = params(predict)
println("before train")
println("loss: $(loss(x_train, y_train))")
println("parameters: $parameters")
for epoch in 1:200
  train!(loss, parameters, data, opt)
end
println("after train")
println("loss: $(loss(x_train, y_train))")
println("parameters: $parameters")

end # module
