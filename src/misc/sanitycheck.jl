using Distributions
using LinearAlgebra


h = 1e-4
dim = 4
N = 30
x = rand(dim) .+ .1
y = rand(dim)
u = rand(dim)
X = rand(dim, N)
ys = X[1, :] + 2*sin.(X[2, :])
l̇ = rand()
ẏ = rand(N)

############################## Sanity Check Prompt #############################
println("Sanity Checks for Derivaitves Should Be Close To Machine Epsilon\n")

############################## Kernel Derivatives ##############################
println("Kernel Derivatives Finite Difference Checks")
println("===========================================")

ψ(ρ; ℓ=1.0, σref=1.0) = σref*exp((ρ/ℓ)^2/2)
dψ(ρ; ℓ=1.0, σref=1.0) = ψ(ρ, ℓ=ℓ, σref=σref) * (ρ/(ℓ^2))
d2ψ(ρ; ℓ=1.0, σref=1.0) = (1/ℓ^2)ψ(ρ, ℓ=ℓ, σref=σref) * (1 + ρ^2/ℓ^2)
δψ(ρ; ℓ=1.0, σref=1.0) = ψ(ρ, ℓ=ℓ, σref=σref) * (-ρ^2/ℓ^3)
δdψ(ρ; ℓ=1.0, σref=1.0) = ψ(ρ, ℓ=ℓ, σref=σref)*(-ρ/ℓ^3) * (ρ^2/ℓ^2 + 2)

k(x, y; ℓ=1.0, σref=1.0) = ψ(norm(x-y), ℓ=ℓ, σref=σref)
δk(x, y; ℓ=1.0, σref=1.0) = δψ(norm(x-y), ℓ=ℓ, σref=σref)

function ∇k(x, y; ℓ=1.0, σref=1.0)
    ρ = norm(x-y)
    dψ(ρ, ℓ=ℓ, σref=σref) * (x-y)/ρ
end

function Hk(x, y; ℓ=1.0, σref=1.0)
    r = x-y
    ρ = norm(r)
    (d2ψ(ρ) - dψ(ρ)/ρ)/ρ^2 * r * r' + dψ(ρ) / ρ * I
end

function δ∇k(x, y; ℓ=1.0, σref=1.0)
    ρ = norm(x-y)
    δdψ(ρ, ℓ=ℓ, σref=σref) * (x-y)/ρ
end

ρ = 1.23

dψρ_test = dψ(ρ)
fd_dψρ = (ψ(ρ+h) - ψ(ρ-h)) / 2h
relerr = (dψρ_test - fd_dψρ) / dψρ_test
println("Finite difference check on dψ:  $relerr")

d2ψρ_test = d2ψ(ρ)
fd_d2ψρ = ( dψ(ρ+h) - dψ(ρ-h) ) / 2h
relerr = (d2ψρ_test - fd_d2ψρ) / d2ψρ_test
println("Finite difference check on d2ψ: $relerr")

δψ_test = δψ(ρ, ℓ=1.3)
fd_δψ = (ψ(ρ, ℓ=1.3+h) - ψ(ρ, ℓ=1.3-h)) / 2h
relerr = (δψ_test - fd_δψ) / δψ_test
println("Finite difference check on δψ: $relerr")

δdψ_test = δdψ(ρ, ℓ=1.3)
fd_δdψ = (dψ(ρ, ℓ=1.3+h) - dψ(ρ, ℓ=1.3-h)) / 2h
relerr = (δdψ_test - fd_δdψ) / δdψ_test
println("Finite difference check on δdψ: $relerr")

∇k_test = ∇k(x, y)'*u
fd_∇k_du = (k(x+h*u, y)-k(x-h*u, y)) / 2h
relerr = (∇k_test - fd_∇k_du) / ∇k_test
println("Finite difference check on ∇k: $relerr")

Hk_test = u'*Hk(x, y)*u
fd_Hk_du = u' * (∇k(x+h*u, y) - ∇k(x-h*u, y)) / 2h
relerr = (Hk_test - fd_Hk_du) / Hk_test
println("Finite difference check on Hk: $relerr")

δk_test = δk(x, y, ℓ=1.3)
fd_δk = (k(x, y, ℓ=1.3+h) - k(x, y, ℓ=1.3-h)) / 2h
relerr = (δk_test - fd_δk) / δk_test
println("Finite difference check on δk: $relerr")

δ∇k_test = u' * δ∇k(x, y, ℓ=1.3)
fd_δ∇k = (δk(x+h*u, y, ℓ=1.3) - δk(x-h*u, y, ℓ=1.3)) / 2h
relerr = (δ∇k_test - fd_δ∇k) / δ∇k_test
println("Finite difference check on δ∇k: $relerr")
################################################################################

################################# Kernel Matrix ################################
function kernel_matrix(X, Y; ℓ=1.0, σref=1.0)
    n = size(X)[2]
    m = size(Y)[2]
    K = zeros(n, m)
    for i = 1:n
        for j = 1:m
            K[i, j] = k(X[:, i], Y[:, j], ℓ=ℓ, σref=σref)
        end
    end
    K
end

KXX = kernel_matrix(X, X, ℓ=1.0, σref=1.0)
c = KXX \ ys
################################################################################

############################### Mean Derivatives ###############################
println("\nMean Derivatives Finite Difference Checks")
println("=========================================")

function μ(x, X, c; ℓ=1.0, σref=1.0)
    μx = 0.0
    for i = 1:size(X)[2]
        μx += c[i] * k(x, X[:,i], ℓ=ℓ, σref=σref)
    end
    return μx
end

function ∇μ(x, X, c; ℓ=1.0, σref=1.0)
    ∇μx = zeros(length(x))
    for i = 1:size(X)[2]
        ∇μx += c[i] * ∇k(x, X[:,i], ℓ=ℓ, σref=σref)
    end
    return ∇μx
end

function Hμ(x, X, c; ℓ=1.0, σref=1.0)
    Hμx = zeros(length(x), length(x))
    for i = 1:size(X)[2]
        Hμx += c[i] * Hk(x, X[:,i], ℓ=ℓ, σref=σref)
    end
    return Hμx
end

function δμ(x, X, c, ẏ, l̇; ℓ=1.0, σref=1.0)
    δμx = 0.0
    KXX = kernel_matrix(X, X)
    KXx = kernel_matrix(X, reshape(x, dim, 1))
    d = KXX \ KXx
    
    for i = 1:size(X)[2]
        δμx += c[i] * δk(x, X[:, i]) * l̇ + d[i]*ẏ[i]
        for j = 1:size(X)[2]
            δμx -= d[i] * δk(X[:, i], X[:, j]) * l̇ * c[j] 
        end
    end
    
    return δμx
end

function δ∇μ(x, X, c, ẏ, l̇; ℓ=1.0, σref=1.0)
    δ∇μx = zeros(length(x))
    W = zeros(size(X))
    
    for ndx = 1:size(X)[2]
        W[:, ndx] = ∇k(x, X[:, ndx])
        δ∇μx += δ∇k(x, X[:, ndx]) * c[ndx] * l̇
    end
    
    W /= kernel_matrix(X, X, ℓ=ℓ, σref=σref)
    z = copy(ẏ)
    
    for i = 1:size(X)[2]
        for j = 1:size(X)[2]
            z[i] -= δk(X[:, i], X[:, j]) * l̇ * c[j]
        end
    end
    
    δ∇μx += W*z
    
    return δ∇μx
end

∇μ_test = u' * ∇μ(x, X, c)
fd_∇μ = (μ(x+h*u, X, c) - μ(x-h*u, X, c)) / 2h
relerr = (∇μ_test - fd_∇μ) / ∇μ_test
println("Finite difference check on ∇μ: $relerr")

Hμ_test = u' * Hμ(x, X, c) * u
fd_Hμ = u' * (∇μ(x+h*u, X, c) - ∇μ(x-h*u, X, c)) / 2h
relerr = (Hμ_test - fd_Hμ) / Hμ_test
println("Finite difference check on Hμ: $relerr")

cplus = kernel_matrix(X, X, ℓ=1.0+h*l̇) \ (ys + h*ẏ)
cminus = kernel_matrix(X, X, ℓ=1.0-h*l̇) \ (ys - h*ẏ)
δμ_test = δμ(x, X, c, ẏ, l̇)
fd_δμ = ( μ(x, X, cplus, ℓ=1.0+h*l̇) - μ(x, X, cminus, ℓ=1.0-h*l̇) ) / 2h
relerr = (δμ_test - fd_δμ) / δμ_test
println("Fininte difference check for δμ: $relerr")

δ∇μ_test = u'*δ∇μ(x, X, c, ẏ, l̇)
fd_δ∇μ = ( δμ(x+h*u, X, c, ẏ, l̇) - δμ(x-h*u, X, c, ẏ, l̇) ) / 2h
relerr = (δ∇μ_test - fd_δ∇μ) / δ∇μ_test 
println("Fininte difference check for δ∇μ: $relerr")
################################################################################

######################## Standard Deviation Derivatives ########################
println("\nStandard Deviation Derivatives Finite Difference Checks")
println("=======================================================")
function σ(x, X; ℓ=1.0, σref=1.0)
    KXX = kernel_matrix(X, X, ℓ=ℓ, σref=σref)
    KXx = kernel_matrix(X, reshape(x, length(x), 1), ℓ=ℓ, σref=σref)
    return √(k(x, x) - dot(KXx, KXX \ KXx))
end

function ∇σ(x, X; ℓ=1.0, σref=1.0)
    ∇σx = zeros(length(x))
    KXx = kernel_matrix(X, reshape(x, length(x), 1))
    d = kernel_matrix(X, X) \ KXx
    
    for i = 1:size(X)[2]
        ∇σx += d[i] * ∇k(x, X[:, i], ℓ=ℓ, σref=σref)
    end
    
    ∇σx /= -σ(x, X, ℓ=ℓ, σref=σref)
    
    return ∇σx
end

function Hσ(x, X; ℓ=1.0, σref=1.0)
    Hσx = zeros(length(x), length(x))
    KXx = kernel_matrix(X, reshape(x, length(x), 1))
    d = kernel_matrix(X, X) \ KXx
    
    W = zeros(size(X))
    for col = 1:size(W)[2]
       W[:, col] = ∇k(x, X[:, col]) 
    end
    W /= kernel_matrix(X, X, ℓ=ℓ, σref=σref)
    
    for i = 1:size(X)[2]
        Hσx += Hk(x, X[:, i])*d[i] + ∇k(x, X[:, i])*W[:, i]'
    end
    
    Hσx += ∇σ(x, X)*∇σ(x, X)'
    Hσx ./= -σ(x, X)
    
    return Hσx
end

function δσ(x, X, l̇; ℓ=1.0, σref=1.0)
    δσx = δk(x, x) * l̇
    KXX = kernel_matrix(X, X)
    KXx = kernel_matrix(X, reshape(x, length(x), 1))
    d = KXX \ KXx
    
    for i = 1:size(X)[2]
        δσx -= 2δk(x, X[:, i]) * d[i] * l̇
        for j = 1:size(X)[2]
            δσx += d[i] * d[j] * δk(X[:, i], X[:, j]) * l̇
        end
    end
    
    δσx /= 2σ(x, X)
    return δσx
end

function δ∇σ(x, X, l̇; ℓ=1.0, σref=1.0)
    δ∇σx = δσ(x, X, l̇) * ∇σ(x, X)
    KXX = kernel_matrix(X, X)
    KXx = kernel_matrix(X, reshape(x, length(x), 1))
    d = KXX \ KXx
    
    W = zeros(size(X))
    for ndx = 1:size(X)[2]
        W[:, ndx] = ∇k(x, X[:, ndx])
    end
    W /= kernel_matrix(X, X, ℓ=ℓ, σref=σref)
    
    z0 = zeros(length(x))
    z1 = zeros(size(X)[2])
    z2 = zeros(size(X)[2])
    
    for i = 1:size(X)[2]
        z0 += δ∇k(x, X[:, i]) * d[i] * l̇
        z2[i] = δk(x, X[:, i]) * l̇
        for j = 1:size(X)[2]
            z1[i] += δk(X[:, i], X[:, j]) * d[j] * l̇
        end
    end
    
    δ∇σx += -W*z1 + W*z2 + z0
    δ∇σx /= -σ(x, X)

    return δ∇σx
end

∇σ_test = u'*∇σ(x, X)
fd_dσ_du = ( σ(x+h*u, X) - σ(x-h*u, X) ) / 2h
relerr = (∇σ_test - fd_dσ_du) / ∇σ_test
println("Finite difference check on ∇σ: $relerr")

Hσ_test = u'*Hσ(x, X)*u
fd_d2σ_du2 = u' * ( ∇σ(x+h*u, X) - ∇σ(x-h*u, X) ) / 2h
relerr = (Hσ_test - fd_d2σ_du2) / Hσ_test
println("Finite difference check on Hσ: $relerr")

δσ_test = δσ(x, X, l̇)
fd_δσ_dl = ( σ(x, X; ℓ=1.0+h*l̇) - σ(x, X; ℓ=1.0-h*l̇) ) / (2h)
relerr = (δσ_test - fd_δσ_dl) / δσ_test
println("Finite difference check on δσ: $relerr")

δ∇σ_test = u' * δ∇σ(x, X, l̇)
fd_δ∇σ_dx = ( δσ(x+h*u, X, l̇) - δσ(x-h*u, X, l̇) ) / (2h)
relerr = (δ∇σ_test - fd_δ∇σ_dx) / δ∇σ_test
println("Finite difference check on δ∇σ: $relerr")
################################################################################

################################ Z Derivatives #################################
println("\nZ Derivatives Finite Difference Checks")
println("======================================")

z(x, X, c, f⁺, ξ; ℓ=1.0, σref=1.0) = (1/σ(x, X, ℓ=ℓ, σref=σref)) * (μ(x, X, c, ℓ=ℓ, σref=σref) - f⁺ - ξ)
∇z(x, X, c, f⁺, ξ; ℓ=1.0, σref=1.0) = (1/σ(x, X, ℓ=ℓ, σref=σref)) * (∇μ(x, X, c, ℓ=ℓ, σref=σref) - ∇σ(x, X, ℓ=ℓ, σref=σref)
    * z(x, X, c, f⁺, ξ, ℓ=ℓ, σref=σref)
)
Hz(x, X, c, f⁺, ξ; ℓ=1.0, σref=1.0) = (1/σ(x, X, ℓ=ℓ, σref=σref)) * (
    Hμ(x, X, c, ℓ=ℓ, σref=σref) - Hσ(x, X, ℓ=ℓ, σref=σref)*z(x, X, c, f⁺, ξ, ℓ=ℓ, σref=σref) - 
    ∇σ(x, X, ℓ=ℓ, σref=σref)*∇z(x, X, c, f⁺, ξ, ℓ=ℓ, σref=σref)' - (∇z(x, X, c, f⁺, ξ, ℓ=ℓ, σref=σref)*∇σ(x, X, ℓ=ℓ, σref=σref)')
)
δz(x, X, c, f⁺, ξ, l̇, ẏ; ḟ⁺=0.0, ξ̇=0.0, ℓ=1.0, σref=1.0) = (1/σ(x, X, ℓ=ℓ, σref=σref)) * (
    δμ(x, X, c, ẏ, l̇, ℓ=ℓ, σref=σref) - ḟ⁺ - ξ̇ - δσ(x, X, l̇, ℓ=ℓ, σref=σref)*z(x, X, c, f⁺, ξ, ℓ=ℓ, σref=σref)
)
δ∇z(x, X, c, f⁺, ξ, l̇, ẏ; ḟ⁺=0.0, ξ̇=0.0, ℓ=1.0, σref=1.0) = (1/σ(x, X, ℓ=ℓ, σref=σref)) * (
    δ∇μ(x, X, c, ẏ, l̇, ℓ=ℓ, σref=σref) - δ∇σ(x, X, l̇, ℓ=ℓ, σref=σref)*z(x, X, c, f⁺, ξ, ℓ=ℓ, σref=σref) -
    δσ(x, X, l̇, ℓ=ℓ, σref=σref)*∇z(x, X, c, f⁺, ξ, ℓ=ℓ, σref=σref) - ∇σ(x, X, ℓ=ℓ, σref=σref)*δz(x, X, c, f⁺, ξ, l̇, ẏ, ḟ⁺=ḟ⁺, ξ̇=ξ̇, ℓ=ℓ, σref=σref)
)

f⁺, ξ = [0.0, 0.0]
∇z_test = u' * ∇z(x, X, c, f⁺, ξ)
fd_dz_dx = (z(x+h*u, X, c, f⁺, ξ) - z(x-h*u, X, c, f⁺, ξ)) / 2h
relerr = (∇z_test - fd_dz_dx) / ∇z_test
println("Finite difference check for ∇z: $relerr")

Hz_test = u'*Hz(x, X, c, f⁺, ξ)*u
fd_∇z_du = u' * (∇z(x+h*u, X, c, f⁺, ξ) - ∇z(x-h*u, X, c, f⁺, ξ)) / 2h
relerr = (Hz_test - fd_∇z_du) / Hz_test
println("Finite difference check for Hz again: $relerr")

cplus = kernel_matrix(X, X, ℓ=1.0+h*l̇) \ (ys + h*ẏ)
cminus = kernel_matrix(X, X, ℓ=1.0-h*l̇) \ (ys - h*ẏ)
δz_test = δz(x, X, c, f⁺, ξ, l̇, ẏ)
fd_dz_dl = ( z(x, X, cplus, f⁺, ξ, ℓ=1.0+h*l̇) - z(x, X, cminus, f⁺, ξ, ℓ=1.0-h*l̇) ) / 2h
relerr = (δz_test - fd_dz_dl) / δz_test
println("Finite difference check for δz: $relerr")

δ∇z_test = u' * δ∇z(x, X, c, f⁺, ξ, l̇, ẏ)
fd_δz_dx = ( δz(x+h*u, X, c, f⁺, ξ, l̇, ẏ) - δz(x-h*u, X, c, f⁺, ξ, l̇, ẏ) ) / 2h
relerr = (δ∇z_test - fd_δz_dx) / δ∇z_test
println("Finite difference check for δ∇z: $relerr")