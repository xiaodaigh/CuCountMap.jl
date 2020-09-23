## CuCountMap

`cucountmap` is a faster `countmap` equivalent utilizing CUDA.jl  for `Vector{T}` where `isbits(T)` and `sizeof(T) <= 2`.

### Usage

```julia
v = rand(Int16, 1_000_000_000)

cucountmap(v) # converts v to cu(v) and then run countmap

using CUDA: cu

cuv = cu(v)
countmap(cuv) # StatsBase.countmap is overloaded for CuArrays
```


### Example

```julia
using CUDA
using CuCountMap
using StatsBase: countmap

v = rand(Int16, 100_000_000);

using BenchmarkTools

@benchmark gpu_countmap = cucountmap($v)

# BenchmarkTools.Trial:
#   memory estimate:  4.17 MiB
#   allocs estimate:  190
#   --------------
#   minimum time:     41.275 ms (0.00% GC)
#   median time:      44.494 ms (0.00% GC)
#   mean time:        52.756 ms (0.48% GC)
#   maximum time:     297.796 ms (3.98% GC)
#   --------------
#   samples:          95
#   evals/sample:     1

@benchmark cpu_countmap = countmap($v)
# BenchmarkTools.Trial:
#   memory estimate:  4.17 MiB
#   allocs estimate:  37
#   --------------
#   minimum time:     132.618 ms (0.00% GC)
#   median time:      134.176 ms (0.00% GC)
#   mean time:        134.874 ms (0.19% GC)
#   maximum time:     145.168 ms (6.76% GC)
#   --------------
#   samples:          38
#   evals/sample:     1

cuv = CUDA.cu(v)
@benchmark gpu_countmap2 = countmap(cuv)

# BenchmarkTools.Trial:
#   memory estimate:  4.17 MiB
#   allocs estimate:  97
#   --------------
#   minimum time:     5.472 ms (0.00% GC)
#   median time:      5.768 ms (0.00% GC)
#   mean time:        6.125 ms (3.91% GC)
#   maximum time:     201.707 ms (96.90% GC)
#   --------------
#   samples:          816
#   evals/sample:     1
```
