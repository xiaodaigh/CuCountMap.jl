using CuCountMap
using StatsBase: countmap

v = rand(Int16, 100_000_000)

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