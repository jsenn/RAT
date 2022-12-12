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