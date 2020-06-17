# using Pkg
# Pkg.activate("c:/git/CuCountmap")
using Test
using SortingAlgorithms
using CuCountMap

v = rand(UInt128, 1_000_000);
@time sorted_v = gpuradixsort(v);
@time sorted_v_cpu = sort(v, alg=RadixSort);

@test sorted_v == sorted_v_cpu

v = rand(UInt16, 1_000_000);
@time a = cucountmap(v);
@time aa = countmap(v);
@test a == aa