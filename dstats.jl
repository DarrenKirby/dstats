function mean(x::Array)::Float64
    "Return the arithmetic mean of Array x"
    sum(x) / length(x)
end


function median(x::Array)::Float64
    "Return the median value of Array x"
    x = sort(x)
    if length(x) % 2 == 1
        return x[ceil(Integer, length(x) / 2)]
    end
    i1 = div(length(x),2)
    (x[i1] + x[i1 + 1]) / 2
end


function mode(x::Array)
    """
    Return the mode(s) of Array x

    The mode may have more than one value, and thus, for consistency
    zero or more modes are returned in an Array no matter the length.
    """
    counts = Dict{Any,Int}()
    mv = 1
    for i in x
        if haskey(counts, i)
            c = (counts[i] += 1)
            if c > mv
                mv = c
            end
        else
            counts[i] = 1
        end
    end
    # All items are unique: no mode
    if mv == 1
        return []
    end
    modes = [m for (m, c) in counts if c == mv]
    # All items have same count: no mode
    if length(modes) == length(unique(a))
        return []
    end
    return modes
end


function variance(x::Array, ddof=1)::Float64
    """
    Return the variance of Array x

    The optional ddof argument is delta degrees of freedom. It defaults
    at 1. Leave at the default for sample variance,
    and pass a 0 for population variance
    """
    x = x - mean(x)
    x = x ^ 2
    sum(x) / (length(x) - ddof)
end
