function mean(x::Array)::Float64
    "Return the arithmetic mean of Array x"
    return sum(x) / length(x)
end


function median(x::Array)::Float64
    "Return the median value of Array x"
    x = sort(x)
    if length(x) % 2 == 1
        return x[ceil(Integer, length(x) / 2)]
    end
    i1 = div(length(x),2)
    return (x[i1] + x[i1 + 1]) / 2
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


function variance(x::Array; ddof::Int64=1)::Float64
    """
    Return the variance of Array x

    The optional ddof argument is delta degrees of freedom.
    It defaults at 1. Leave at the default for sample
    variance, and pass a 0 for population variance
    """
    x = x .- mean(x)
    x = x .^ 2
    return sum(x) / (length(x) - ddof)
end


function std(x::Array; ddof::Int64=1)::Float64
    "Return the standard deviation of Array x"
    return sqrt(variance(x, ddof=ddof))
end


function fano_factor(x::Array)::Float64
    "Return the Fano factor of Array x"
    m = mean(x)
    if m == 0
        throw(error("Cannot calculate Fano factor of mean-centered data"))
    end
    return variance(x) / m
end


function coefficient_of_variation(x::Array)::Float64
    "Return the coefficient of variation of Array x"
    m = mean(x)
    if m == 0
        throw(error("Cannot calculate COV of mean-centered data"))
    end
    return std(x) / m
end


function skew(x::Array)::Float64
    "Return the skew aka third moment of sequence x."
    arr = (x .- mean(x))
    arr = arr .^ 3
    return sum(arr) / (length(arr) * std(x, ddof=0) ^ 3)
end


function kurtosis(x::Array; fisher::Bool=true)::Float64
    "Return the kurtosis aka fourth moment of sequence x."
    arr = (x .- mean(x))
    arr = arr .^ 4
    arr = sum(arr) / (length(arr) * std(x, ddof=0) ^ 4)
    return (fisher==true) ? arr - 3.0 : arr
end


function moments(x::Array)::Array
    "Return the first four statistical moments in an Array"
    return [mean(x), variance(x), skew(x), kurtosis(x)]
end


# Aliases
moment1 = mean
moment2 = variance
moment3 = skew
moment4 = kurtosis
