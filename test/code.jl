# using Pkg
# Pkg.activate("c:/git/CuCountmap")

using SortingAlgorithms
using CuCountMap

v = rand(UInt128, 100_000_000);
@time sorted_v = gpuradixsort(v);
@time sorted_v_cpu = sort(v, alg=RadixSort);

@test sorted_v == sorted_v_cpu

using CuCountmap

@time a = cucountmap(v);
@time aa = countmap(cpuv);
@test a == aa