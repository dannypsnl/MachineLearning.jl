module layers

using Flux

model = Chain(
  Dense(10, 5, σ),
  Dense(5, 2),
  softmax)

println(model(rand(10)))

end