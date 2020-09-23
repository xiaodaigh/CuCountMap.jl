# Weave readme
using Pkg
cd("c:/git/CuCountMap")
Pkg.activate("c:/git/CuCountMap/readme-env")

using Weave

weave("README.jmd", out_path=:pwd, doctype="github")


if false
    tangle("README.jmd")
end
