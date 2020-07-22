abstract type KernelFunction <: Number end

include("squared_exponential.jl")

function kernel_matrix(X, Y, kernel::KernelFunction)
    n = size(X)[2]
    m = size(Y)[2]
    K = zeros(n, m)
    for i = 1:n
        for j = 1:m
            K[i, j] = kernel(X[:, i], Y[:, j])
        end
    end
    K
end