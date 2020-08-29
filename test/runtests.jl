# using Pkg
# Pkg.activate("c:/git/CuCountmap")
using CuCountMap
using SortingAlgorithms
using Test

@testset "CuCountMap.jl - countmap" begin
    v = rand(UInt16, 1_000_000);
    @time a = cucountmap(v);
    @time aa = countmap(v);
    @test a == aa
end

@testset "CuCountMap.jl - sort" begin
    v = rand(UInt64, 1_000_000);
    @time sorted_v = gpuradixsort(v);
    @time sorted_v_cpu = sort(v, alg=RadixSort);
    @test sorted_v == sorted_v_cpu
end
