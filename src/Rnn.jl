module Rnn

using Flux
using Flux: train!, reset!
using Flux.Losses: mse

m = Chain(
  RNN(2, 5),
  Dense(5, 1))

function loss(x, y)
  reset!(m)
  sum(mse(m(xi), yi) for (xi, yi) in zip(x, y))
end

seq_init = [rand(Float32, 2)]
seq_1 = [rand(Float32, 2) for i = 1:3]
seq_2 = [rand(Float32, 2) for i = 1:3]

y1 = [rand(Float32, 1) for i = 1:3]
y2 = [rand(Float32, 1) for i = 1:3]

X = [seq_1, seq_2]
Y = [y1, y2]
data = zip(X, Y)

ps = params(m)
opt = ADAM(1e-3)

@show(loss(seq_1, y1))
for epoch in 1:20
  train!(loss, ps, data, opt)
  @show(loss(seq_1, y1))
end

end
