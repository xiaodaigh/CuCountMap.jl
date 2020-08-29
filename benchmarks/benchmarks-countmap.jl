using Pkg
Pkg.activate("c:/git/CuCountmap")

using CuArrays
CuArrays.allowscalar(false)
using CUDAnative

using CUDAnative: atomic_add!

using StatsBase
import StatsBase: countmap

function cucountmap!(buffer, v)
    i = (blockIdx().x - 1) * blockDim().x + threadIdx().x
    stride = gridDim().x * blockDim().x

    for j = i:stride:length(v)
        b = Int(v[j]) + Int(32769)
        @atomic buffer[b] = buffer[b] + UInt32(1)
    end
    return
end

function countmap(v::CuArray{T, N, NN}; threads = 256, blocks = 1024) where {T, N, NN}
    buffer = CuArrays.zeros(UInt32, 2^16)
    CuArrays.@sync @cuda threads = threads blocks = blocks cucountmap!(buffer, v)
    buffer
end

function cucountmap2!(buffer, v)
    i = (blockIdx().x - 1) * blockDim().x + threadIdx().x
    stride = gridDim().x * blockDim().x

    for j = i:stride:length(v)
         b = Int(v[j]) + Int(32769)
         @atomic buffer[b, blockIdx().x] = buffer[b, blockIdx().x] + UInt32(1)
    end

    return
end

function countmap2(v::CuArray{T, N, NN}; threads = 256, blocks = 1024) where {T, N, NN}
    buffer = CuArrays.zeros(UInt32, 2^16, blocks)
    CuArrays.@sync @cuda threads = threads blocks = blocks cucountmap2!(buffer, v)

    for i in 2:blocks
        buffer[:, 1] .+= buffer[:, i]
    end
    buffer[:, 1]
end

v = CuArray(rand(Int16, 100_000_000))

################################################################
# basic check v2
################################################################

buffer = CuArrays.zeros(UInt32, 2^16, 1024)

CuArrays.@sync @cuda threads = 512 blocks = 1024 cucountmap2!(buffer, v)

@time countmap2(v)
@time countmap(v)
vc = collect(v)

################################################################
# basic check
################################################################

buffer = CuArrays.zeros(UInt32, 2^16)

CuArrays.@sync @cuda threads = 512 blocks = 1024 cucountmap!(buffer, v)

countmap(v)
vc = collect(v)

################################################################
# detail testing
################################################################
res = @time countmap(collect(v))

buffer_check = zeros(Int, 2^16)
for (k, i) in collect(res)
    buffer_check[k+32769] = i
end

collect(buffer) == buffer_check

#@device_code_warntype cucountmap!(buffer, v)

################################################################
# benchmark
################################################################
using BenchmarkTools
vc = collect(v)

@benchmark countmap($vc)
@benchmark countmap($v)
