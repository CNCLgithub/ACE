module ACE

using Gen
using Luxor
using GenRFS
using PythonCall
using StaticArrays
using DataStructures
using NearestNeighbors
using DocStringExtensions

# Pointers to Python imports
const sys = Ref{Py}()
const jax_script = Ref{Py}()

function __init__()
    # Import jax script
    py_sys = pyimport("sys")
    py_dir = normpath(joinpath(@__DIR__, "..", "python"))
    if !(py_dir in py_sys.path)
        py_sys.path.append(py_dir)
    end
    jax_script[] = pyimport("jax_script")
end

include("utils/utils.jl")
include("python.jl")

end # module ACE
