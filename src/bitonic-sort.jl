"""Attempting to implement bitonic sort"""

using CUDA

# sort kernel
bisort!(shared) = begin
  k = UInt16(2)
  NUM = length(cushared)
  while (k <= NUM)
    j = div(k, 2)
    while j >= 1
      tid = UInt(((blockIdx().x - 1) * blockDim().x + threadIdx().x)-1)
      ixj = tid⊻UInt(j)
      if ixj > tid
        if (tid & k) == 0
          if shared[tid+1] > shared[ixj+1]
            tmp = shared[ixj+1]
            shared[ixj+1] = shared[tid+1]
            shared[tid+1] = tmp
          end
        else
          if shared[tid+1] < shared[ixj+1]
            tmp = shared[ixj+1]
            shared[ixj+1] = shared[tid+1]
            shared[tid+1] = tmp
          end
        end
      end
    end
    k *= 2
  end
  return
end

bitonicsort!(cushared, NUM) = begin
  nblocks = ceil(Int, NUM/256)
  @cuda threads = 256 blocks = 1024 bisort!(cushared)
end


using SortingAlgorithms, BenchmarkTools
shared = rand(Float32, 2^26)
cpusort = @belapsed sort!($shared, alg=RadixSort) #0.788

shared = rand(Float32, 2^26)

measure_gpu_sort(shared) = begin
  res = Float64[]
  for i = 1:3
    cushared = cu(shared)
    # sorted_shared = sort(shared, alg=RadixSort)
    # println("exp false;")
    # println("got $(collect(cushared) |> issorted)")
    t = Base.@elapsed begin
      bitonicsort!(cushared, length(shared))
      CUDA.synchronize()
    end
    # xx = collect(cushared)
    # println("exp true;")
    # println("got $(xx |> issorted); max error: $(1_000_000_000maximum(xx .- sorted_shared))")
    # push!(res, t)
  end

  res
end

@time measure_gpu_sort(shared)
# 6.457073086
# 2.774147852
# 2.771599214
# 2.770980271
# 2.778133025
# 2.769555927
# 2.799603755
# 2.774497496
# 2.790657341
# 2.790034242
