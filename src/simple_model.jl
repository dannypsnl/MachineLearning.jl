module simple_model

using Flux

W = rand(2, 5)
b = rand(2)

predict(x) = W*x .+ b

function loss(x, y)
  y1 = predict(x)
  sum((y .- y1).^2)
end

x, y = rand(5), rand(2)
println("loss: $(loss(x, y))")

gs = gradient(() -> loss(x, y), params(W, b))
W1 = gs[W]
W .-= 0.1 .* W1
println("loss: $(loss(x, y))")

end