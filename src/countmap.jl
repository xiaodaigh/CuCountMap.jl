export countmap, cucountmap, cucountmap!

using CUDA
using StatsBase
import StatsBase: countmap

cucountmap(v) = cu(v) |> countmap

# The winner of bencharmks/bencharmks-countmap
function cucountmap!(buffer, v::CuDeviceArray{T, 1, NN2}) where {T, NN2} #(buffer::CuArray{UInt32, 1, NN1}, v::CuArray{T, 1, NN2}) where {NN1, T, NN2}
    i = (blockIdx().x - 1) * blockDim().x + threadIdx().x
    stride = gridDim().x * blockDim().x

    addnum = 1 - Int(typemin(T))
    for j = i:stride:length(v)
        b = Int(v[j]) + addnum
        @atomic buffer[b] = buffer[b] + 1
    end
    return
end

function countmap(v::CuArray{T}; threads = 256, blocks = 1024) where {T}
    st = sizeof(T)
    @assert st <= 2
    buffer = CUDA.zeros(Int, 2^(8st))
    CUDA.@sync @cuda threads = threads blocks = blocks cucountmap!(buffer, v)
    values = typemin(T) : typemax(T)
    Dict(zip(values, collect(buffer)))
end
