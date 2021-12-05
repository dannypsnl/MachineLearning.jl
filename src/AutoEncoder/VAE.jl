using MLDatasets
using Flux
using Flux: train!, @epochs, mse, throttle, params
using ConditionalDists
using ConditionalDists: SplitLayer
using GenerativeModels

train_x, _ = MNIST.traindata()
train_x = reshape(train_x, :, size(train_x, 3)) |> gpu
data = Flux.Data.DataLoader(train_x, batchsize=200, shuffle=true)

encoder = Chain(
  Dense(size(flat_x, 1), 512, relu),
  Dense(512, 256, relu),
  SplitLayer(256, [2, 2], [identity, softplus])
) |> ConditionalMvNormal
decoder = Chain(
  Dense(2, 256, relu),
  Dense(256, 512, relu),
  SplitLayer(512, [size(flat_x, 1), 1], σ)
) |> ConditionalMvNormal
model = VAE(zlength, encoder, decoder) |> gpu

loss(x) = -elbo(model,x)
ps = params(model)
opt = ADAM()

@epochs 50 begin
  @info "Epoch $e" loss(flat_x)
  Flux.train!(loss, ps, data, opt)
end
