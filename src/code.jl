
using Pkg
Pkg.activate("c:/git/CuCountmap")

using SortingAlgorithms
using CuCountmap

v = rand(UInt128, 100_000_000);
@time sorted_v = gpuradixsort(v);
@time sorted_v_cpu = sort(v, alg=RadixSort);

sorted_v == sorted_v_cpu



hist = collect(radixhist(cu(v)))



v = rand(UInt16, 100_000_000)
@time countmap(v);

using CUDA
cuv = cu(v)
@time countmap(cuv);



sorted_v_cpu == sorted_v

issorted(sorted_v)


@time cumsum(v)
cpuv = collect(v)
@time cumsum(cpuv)


using CuCountmap

@time a = cucountmap(v);
@time aa = countmap(cpuv);
a == aa

cpuv = collect(v)



a == aa



using SortingLab
vc=collect(v)

@time fsort(vc)
using SortingAlgorithms
@time sort(vc, alg=RadixSort)