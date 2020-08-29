# using Pkg
# Pkg.activate("c:/git/CuCountmap")
using SortingAlgorithms
using CuCountMap

v = rand(UInt64, 100_000_000);

@benchmark sorted_v = gpuradixsort($v)
# BenchmarkTools.Trial:
#   memory estimate:  1.49 GiB
#   allocs estimate:  2550
#   --------------
#   minimum time:     3.351 s (0.10% GC)
#   median time:      3.404 s (1.82% GC)
#   mean time:        3.404 s (1.82% GC)
#   maximum time:     3.457 s (3.48% GC)
#   --------------
#   samples:          2
#   evals/sample:     1


@benchmark sorted_v_cpu = sort($v, alg=RadixSort)
# BenchmarkTools.Trial:
#   memory estimate:  1.49 GiB
#   allocs estimate:  18
#   --------------
#   minimum time:     3.648 s (0.08% GC)
#   median time:      3.708 s (1.75% GC)
#   mean time:        3.708 s (1.75% GC)
#   maximum time:     3.767 s (3.36% GC)
#   --------------
#   samples:          2
#   evals/sample:     1