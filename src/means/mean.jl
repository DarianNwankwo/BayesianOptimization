abstract type MeanFunction <: Number end

struct ZeroMean <: MeanFunction
    dim::Int
    ZeroMean(dim) = new(dim)
end
(m::ZeroMean)() = zeros(m.dim)