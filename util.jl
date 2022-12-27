using Distributions

"""
    partialsum(xs)

Return an array `ps` where `ps[i] = sum(xs[1:i])`
"""
function partialsum(xs)
    if isempty(xs)
        return []
    end
    ps = similar(xs)
    ps[1] = xs[1]
    for i = 2:lastindex(xs)
        ps[i] = xs[i] + ps[i-1]
    end
    return ps
end

"""
    fdiff(xs::Array)

Returns an array of length `length(xs) - 1` whose `i`th entry is `x[i+1] - x[i]`.
This is (almost) the inverse of `partialsum`.
"""
function fdiff(xs::Array)
    if length(xs) < 2
        return []
    end

    res = zeros(eltype(xs), length(xs) - 1)
    xcurr = xs[1]
    for i = 2:length(xs)
        res[i-1] = xs[i] - xcurr
        xcurr = xs[i]
    end

    return res
end

"""
    weightedmedian(xs; residual=false)

Return the index `imin` where the residual `sum(xs[i:end]) - sum(xs[1:i])`
is smallest, but still greater than 0. This is a discrete analogue of the
median of a probability density function.

If `residual` is `false`, returns `imin`; if `true`, returns `(imin, residual)`
"""
function weightedmedian(xs; residual=false)
    ps = partialsum(xs)
    resmin = Inf
    imin = 0
    # TODO: could do binary search here?
    for i = 1:lastindex(ps)
        below = ps[i] - xs[i]
        above = ps[end] - (i==1 ? 0 : ps[i-1])
        res = above - below
        if res < 0
            break
        elseif res < resmin
            resmin = res
            imin = i
        end
    end
    @assert(imin != 0 && isfinite(resmin))
    return residual ? (imin, resmin) : imin
end

function normalize!(xs::Array)
    if isempty(xs)
        return
    end
    m = minimum(xs)
    M = maximum(xs)
    if m == M
        fill!(xs, 0)
    else
        map!(x->(x-m)/(M-m), xs, xs)
    end
end

function normalize_audio!(xs::Array{T}) where T<:AbstractFloat
    if isempty(xs)
        return
    end

    m = minimum(xs)
    M = maximum(xs)
    if m == M
        fill!(xs, T(0.))
    else
        s = LinearScale(m, M, T(-1.), T(1.))
        map!(s, xs, xs)
    end
end

struct LinearScale{T<:AbstractFloat}
    fromlow::T
    fromhigh::T
    tolow::T
    tohigh::T
end

function evaluate(s::LinearScale{T}, t::T) where T<:AbstractFloat
    param = (t-s.fromlow)/(s.fromhigh-s.fromlow)
    return s.tolow + param * (s.tohigh - s.tolow)
end

function invert(s::LinearScale{T}, t::T) where T<:AbstractFloat
    param = (t-s.tolow)/(s.tohigh-s.tolow)
    return s.fromlow + param * (s.fromhigh - s.fromlow)
end

function (s::LinearScale{T})(t::T) where T<:AbstractFloat
    evaluate(s, t)
end

function bernoulli(p::Number)
    Int(rand() < p)
end

function gaussian(u = 0, s = 1)
    return rand(Normal(u,s))
end