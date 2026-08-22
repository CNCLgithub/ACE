
"""
$(TYPEDSIGNATURES)

Computes softmax of an array, with temperature `t`.
"""
function softmax(x, t::Float64 = 1.0)
    n = length(x)
    out = Vector{Float64}(undef, n)
    softmax!(out, x, t)
    return out
end

function softmax!(out::Array{Float64}, x, t::Float64 = 1.0)
    nx = length(x)
    maxx = maximum(x)
    sxs = 0.0

    if maxx == -Inf
        out .= 1.0 / nx
        return nothing
    end

    @inbounds for i = 1:nx
        out[i] = @fastmath exp((x[i] - maxx) / t)
        sxs += out[i]
    end
    # rmul!(out, 1.0 / sxs)
    map!(x -> x / sxs, out)
    return nothing
end


# function welfords_stats(v::Array{Float64})
#     n = length(v)
#     mean = 0.0
#     vari = 0.0
#     sosq = 0.0
#     @inbounds for i = 1:n
#         x = exp(v[i]) # REVIEW: underflow?
#         prev_mean = mean
#         mean += (x - mean) / n
#         sosq += (x - mean) * (x - prev_mean)
#     end
#     variance = sosq / (n-1)
#     sdev = sqrt(variance)

#     (mean, sdev)
# end

# function exp_log_ratio(x::Float64, y::Float64)
#     lx = log(x + 1E-4)
#     ly = log(y + 1E-4)
#     exp(lx - logsumexp(lx, ly))
# end

# """
#     kld_bernoulli(p::Float64, q::Float64) -> Float64

# Compute the KL divergence KL(P || Q) between two Bernoulli distributions
# with success probabilities p and q.

# KL(P || Q) = p * log(p/q) + (1-p) * log((1-p)/(1-q))

# Handles edge cases where p = 0 or p = 1 using the convention 0 * log(0) = 0.
# Throws an error if q = 0 or q = 1 but p ≠ q (undefined / infinite divergence).
# """
# function kld_bernoulli(p::Float64, q::Float64)::Float64
#     # Validate inputs
#     0.0 ≤ p ≤ 1.0 || throw(ArgumentError("p must be in [0, 1], got $p"))
#     0.0 ≤ q ≤ 1.0 || throw(ArgumentError("q must be in [0, 1], got $q"))

#     # If Q has zero mass where P has positive mass, KL is infinite
#     (q == 0.0 && p > 0.0) && return Inf
#     (q == 1.0 && p < 1.0) && return Inf

#     # Use xlogy(a, b) = a * log(b/a), treating 0 * log(0) = 0
#     term1 = p == 0.0 ? 0.0 : p * log(p / q)
#     term2 = (1 - p) == 0.0 ? 0.0 : (1 - p) * log((1 - p) / (1 - q))

#     return term1 + term2
# end

# function logit(x::Float64)
#     x = clamp(x, 0.001, 0.999)
#     log(x/(1-x))
# end

function norm(d::S2V)
    x, y  = d
    sqrt(x^2 + y^2)
end

# adapted from: https://github.com/yp-mit/spectre/blob/master/lossless-sc/my_logdiffexp.m
function logabsdiffexp(a,b)
# % Self written function to compute the log difference of the exponentials of a, b.
# % This function avoids overflow in computing the exponentials.
# % Input: 
# % a, b: two values
# % Output:
# % log_x: log of the difference of the exponentials of a, b
    if b > a
        (a, b) = (b, a)
    end
    a + log(1-exp(b-a))
end
