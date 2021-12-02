module layers

using Flux

model = Chain(
  Dense(10, 5, σ),
  Dense(5, 2),
  softmax)

@show(model(rand(10)))

end
