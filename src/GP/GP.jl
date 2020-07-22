mutable struct GP
    X::Array{Float64, 2} # initial sample locations (n x d)
    Y::Array{Float64, 1} # initial sample values (n x 1)
    mean::MeanFunction
    kernel::KernelFunction
    kernel_matrix::Array{Float64, 2}

    function GP(X, Y, mean, kernel)
        n = size(X)[2]
        gp = new(X, Y, mean, kernel, zeros(n, n))
        return gp
    end
end

function kernel_matrix!(gp::GP)
    n = size(gp.X)[2]
    for i = 1:n
        for j = 1:n
            ρ = 
            gp.kernel_matrix[i, j] = gp.kernel(gp.X[:, i], gp.X[:, j])
        end
    end
end