################################################################################
# The naming convention throughout this document will follow the given scheme:
# - <kernel_name>
# Each kernel function should implement a callable that evaluates the kernel with the 
# respective hyperparamters
################################################################################
struct SquaredExponential <: KernelFunction
    ℓ
    σ
    SquaredExponential(ℓ=1.0, σ=1.0) = new(ℓ, σ)
end

# Add a big comment describing all of this
ψ(ρ; ℓ=1.0, σref=1.0) = σref*exp((ρ/ℓ)^2/2)
dψ(ρ; ℓ=1.0, σref=1.0) = ψ(ρ, ℓ=ℓ, σref=σref) * (ρ/(ℓ^2))
d2ψ(ρ; ℓ=1.0, σref=1.0) = (1/ℓ^2)ψ(ρ, ℓ=ℓ, σref=σref) * (1 + ρ^2/ℓ^2)
# Perturbations to hypers
δψ(ρ; ℓ=1.0, σref=1.0) = ψ(ρ, ℓ=ℓ, σref=σref) * (-ρ^2/ℓ^3)
δdψ(ρ; ℓ=1.0, σref=1.0) = ψ(ρ, ℓ=ℓ, σref=σref)*(-ρ/ℓ^3) * (ρ^2/ℓ^2 + 2)

function (se::SquaredExponential)(x, y)
    ρ = norm(x-y)
    ψ(ρ, ℓ=se.ℓ, σref=se.σ)
end

function gradient(se::SquaredExponential, x, y) 
    ρ = norm(x-y)
    dψ(ρ, ℓ=se.ℓ, σref=se.σ) * (x-y)/ρ 
end

function hessian(se::SquaredExponential, x, y)
    r = x-y
    ρ = norm(x-y)
    (d2ψ(ρ, ℓ=se.ℓ, σref=se.σ) - dψ(ρ, ℓ=se.ℓ, σref=se.σ)/ρ)/ρ^2 * r * r' + dψ(ρ, ℓ=se.ℓ, σref=se.σ)/ρ * I
end

hypersderiv(se::SquaredExponential, x, y) = dψ(norm(x-y), ℓ=se.ℓ, σref=se.σ)

function mixedderiv(se::SquaredExponential, x, y)
    ρ = norm(x-y)
    δdψ(ρ, ℓ=se.ℓ, σref=se.σ) * (x-y)/ρ
end