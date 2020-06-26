################################################################################
# The naming convention throughout this document will follow the given scheme:
# - <kernel_name>
################################################################################

# Spatial derivatives of the squared exponential kernel
SE(ρ; ℓ=1.0, σref=1.0) = σref*exp((ρ/ℓ)^2/2)
dSE(ρ; ℓ=1.0, σref=1.0) = SE(ρ, ℓ=ℓ, σref=σref) * (ρ/(ℓ^2))
d2SE(ρ; ℓ=1.0, σref=1.0) = (1/ℓ^2)SE(ρ, ℓ=ℓ, σref=σref) * (1 + ρ^2/ℓ^2)

# Derivatives of the squared exponential kernel w.r.t. kernely hyperparameter ℓ
δSE(ρ; ℓ=1.0, σref=1.0) = SE(ρ, ℓ=ℓ, σref=σref) * (-ρ^2/ℓ^3)

# Mixed derivative of the squared exponential kernel
δdSE(ρ; ℓ=1.0, σref=1.0) = SE(ρ, ℓ=ℓ, σref=σref)*(-ρ/ℓ^3) * (ρ^2/ℓ^2 + 2)