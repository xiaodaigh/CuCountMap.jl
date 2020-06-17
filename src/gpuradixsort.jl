export gpuradixsort, gpuradixsort!

using CUDA
CUDA.allowscalar(false)

const RADIX_SIZE = 11
const RADIX_MASK = UInt16(2^RADIX_SIZE-1)

# The winner of bencharmks/bencharmks-countmap
function radixhist!(buffer, v::CuDeviceArray{T, 1, NN2}) where {T, NN2}
    i = (blockIdx().x - 1) * blockDim().x + threadIdx().x
    stride = gridDim().x * blockDim().x

    for k in 1:ceil(Int, 8sizeof(T) / RADIX_SIZE)
        for j = i:stride:length(v)
            bits_to_shift = (k-1)*RADIX_SIZE
            vj_shifted = v[j] >> bits_to_shift
            b = Int(vj_shifted & RADIX_MASK) + 1
            @atomic buffer[b, k] = buffer[b, k] + 1
        end
    end
    return
end

# count
function radixhist(v::CuArray{T, N, NN}; threads = 256, blocks = 1024) where {T, N, NN}
    buffer = CUDA.zeros(Int, 2^RADIX_SIZE, ceil(Int, 8sizeof(T) / RADIX_SIZE))
    @sync @cuda threads = threads blocks = blocks radixhist!(buffer, v)
    for i in 1:ceil(Int, 8sizeof(T) / RADIX_SIZE)
        buffer[:, i] = cumsum(buffer[:, i])
    end
    buffer
end

gpuradixsort(v) = gpuradixsort!(copy(v))

function gpuradixsort!(vs::AbstractVector{T}) where T
    bin = collect(radixhist(cu(vs)))
    ts = similar(vs)

    # use the histogram to sort the data
    hi = length(vs)
    lo = 1
    len = hi-lo+1
    iters = ceil(Int, 8sizeof(T) / RADIX_SIZE)
    for j = 1:iters
        # Unroll first data iteration, check for degenerate case
        v = vs[hi]
        idx = Int((v >> ((j-1)*RADIX_SIZE)) & RADIX_MASK) + 1

        cbin = bin[:,j]
        ci = cbin[idx]
        ts[ci] = vs[hi]
        cbin[idx] -= 1

        # Finish the loop...
        @inbounds for i in hi-1:-1:lo
            v = vs[i]
            idx = Int((v >> ((j-1)*RADIX_SIZE)) & RADIX_MASK) + 1
            ci = cbin[idx]
            ts[ci] = vs[i]
            cbin[idx] -= 1
        end
        vs,ts = ts,vs
    end

    vs
end
