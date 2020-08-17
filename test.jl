using Flux
using Flux: swish
# the neural networks
comp = 8
nx = 8
nh = 8
nz = 8


A = [Chain(Dense(nx, nh, swish)) for i in 1:comp]
#A = Chain(Dense(nx, nh, swish))
ps = Flux.params(A);

# the whole loss function
function loss(x, y)
    temp_loss = [sum(A[i](x) .- y) for i in 1:comp]
    #temp_loss = A[1](x)
    sum(temp_loss)
end

ps = Flux.params(A[1]);

x = rand(comp)
y = similar(x)

opt = ADAM()

Flux.train!(loss, ps, (x,y) , opt)
