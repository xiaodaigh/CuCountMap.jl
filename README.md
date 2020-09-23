## CuCountmap

`cucountmap` is a faster `countmap` equivalent utilizing CUDA.jl  for `Vector{T}` where `isbits(T)` and `sizeof(T) <= 2`.

### Usage

````julia

using CuCountMap

v = rand(Int16, 1_000_000)

cucountmap(v) # converts v to cu(v) and then run countmap

using CUDA: cu

cuv = cu(v)
countmap(cuv) # StatsBase.countmap is overloaded for CuArrays
````


````
Dict{Int16,Int64} with 65536 entries:
  -13838 => 17
  22035  => 12
  -15285 => 19
  -13843 => 12
  -18190 => 19
  -20309 => 11
  19698  => 11
  -8633  => 20
  -17455 => 12
  -16936 => 22
  29981  => 14
  -20376 => 15
  7237   => 20
  -27415 => 10
  17959  => 17
  27248  => 17
  -32758 => 17
  -13400 => 17
  5784   => 10
  ⋮      => ⋮
````






### Example & Benchmarks

````julia

using CUDA
using CuCountMap
using StatsBase: countmap

v = rand(Int16, 10_000_000);

using BenchmarkTools

cpu_to_gpu_benchmark = @benchmark gpu_countmap = cucountmap($v)
````


````
BenchmarkTools.Trial: 
  memory estimate:  4.17 MiB
  allocs estimate:  76
  --------------
  minimum time:     4.751 ms (0.00% GC)
  median time:      4.974 ms (0.00% GC)
  mean time:        5.320 ms (3.50% GC)
  maximum time:     14.950 ms (55.27% GC)
  --------------
  samples:          940
  evals/sample:     1
````



````julia

cpu_to_cpu_benchmark = @benchmark cpu_countmap = countmap($v)
````


````
BenchmarkTools.Trial: 
  memory estimate:  4.17 MiB
  allocs estimate:  37
  --------------
  minimum time:     14.915 ms (0.00% GC)
  median time:      15.344 ms (0.00% GC)
  mean time:        15.670 ms (1.06% GC)
  maximum time:     22.093 ms (28.90% GC)
  --------------
  samples:          320
  evals/sample:     1
````



````julia

cuv = CUDA.cu(v)
gpu_to_gpu_benchmark = @benchmark gpu_countmap2 = countmap(cuv)
````


````
BenchmarkTools.Trial: 
  memory estimate:  4.17 MiB
  allocs estimate:  64
  --------------
  minimum time:     2.512 ms (0.00% GC)
  median time:      2.692 ms (0.00% GC)
  mean time:        2.984 ms (5.91% GC)
  maximum time:     17.421 ms (73.12% GC)
  --------------
  samples:          1675
  evals/sample:     1
````





#### Benchmark Plot

````julia

using Plots
using Statistics: mean

cpu_to_gpu = mean(cpu_to_gpu_benchmark.times)/1000/1000
gpu_to_gpu = mean(gpu_to_gpu_benchmark.times)/1000/1000
cpu_to_cpu = mean(cpu_to_cpu_benchmark.times)/1000/1000

plot(
["CPU Array on CPU \n countmap(v)", "convert CPU Array to GPU array on GPU \n cucountmap(cu(v))", "GPU array on GPU \n cucountmap(cuv)"],
[cpu_to_cpu, cpu_to_gpu, gpu_to_gpu],
seriestypes = :bar, title="CuCountMap.cucountmap vs StatsBase.countmap", label="ms",
legendtitle="Mean time")
````


![](figures/README_5_1.png)
