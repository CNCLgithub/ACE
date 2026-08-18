"""
    $TYPEDEF

A simple Fill array implementation.
Supports:

- length
- getindex
- eltype
"""
struct Fill{T}
    val::T
    n::Int
end

import Base.length
Base.length(a::Fill) = a.n

import Base.getindex
Base.getindex(a::Fill{T}, ::Int) where {T} = a.val

import Base.eltype
Base.eltype(a::Fill{T}) where {T} = T
