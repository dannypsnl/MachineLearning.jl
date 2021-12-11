# Basic idea

機器學習的核心是：假設有一個函數 $f$ 可以解決問題，我們要試圖找出函數 $f'$ 盡可能的跟 $f$ 輸出接近

現在假設有一個函數 $actual(x) = 4x + 2$，我們用 Julia 寫成

```julia
actual(x) = 4x + 2
```

接著我們要有訓練資料跟驗證資料，在現在的簡單情境下，我們可以用 `actual` 函數生成結果

```julia
using Flux: Data.DataLoader

x_train, x_test = [0 1 2 3 4 5], [6 7 8 9 10]
y_train, y_test = actual.(x_train), actual.(x_test)
data = DataLoader((x_train, y_train), shuffle=true)
```

接著我們建立模型跟誤差函數，模型就是 $f'$，誤差函數的意思則是 $f'(x)$ 的結果跟真實的 $y$ 的差距。誤差函數有很多計算方式，這裡我們用 [MSE](https://zh.wikipedia.org/wiki/%E5%9D%87%E6%96%B9%E8%AF%AF%E5%B7%AE) 函數。`Dense(1, 1)` 表示輸入為一個，輸出為一個的函數，由於有 Weight 跟 Bias，我們的 model 正好有兩個參數，我們最後會看到這兩個參數經過訓練之後會趨近 4 跟 2

```julia
using Flux

model = Dense(1, 1)
loss(x, y) = Flux.Losses.mse(model(x), y)
```

接著我們要選擇**學習優化器 optimizer**(也有拼作 optimiser 的)，這裏我們選用 stochastic gradient decent，他是往參數梯度的方向去更新權重，參數可以用 `params(model)` 得到。最後就是放到 `@epochs` 裡面去跑

```julia
using Flux: train!, @epochs

opt = Descent()

@epochs 40 begin
  train!(loss, params(model), data, opt)
  @show(loss(x_test, y_test))
  @show(params(model))
end
```

我們把參數跟誤差用 `@show` 印出來，可以得到以下結果

```
[ Info: Epoch 1
loss(x_test, y_test) = 62.672203f0
params(model) = Params([Float32[4.9183693;;], Float32[2.46236]])
[ Info: Epoch 2
loss(x_test, y_test) = 134.20432f0
params(model) = Params([Float32[2.2584872;;], Float32[4.6122694]])
[ Info: Epoch 3
loss(x_test, y_test) = 204.82014f0
params(model) = Params([Float32[5.521385;;], Float32[3.977801]])
...
[ Info: Epoch 38
loss(x_test, y_test) = 1.6675525f-5
params(model) = Params([Float32[4.0004225;;], Float32[2.0006583]])
[ Info: Epoch 39
loss(x_test, y_test) = 1.2968558f-6
params(model) = Params([Float32[4.000121;;], Float32[2.000157]])
[ Info: Epoch 40
loss(x_test, y_test) = 1.7670478f-6
params(model) = Params([Float32[4.0001535;;], Float32[2.000083]])
```

可以看到最後參數確實是接近 4 跟 2 的
