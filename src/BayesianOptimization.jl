module BayesianOptimization

using LinearAlgebra

# Write your package code here.
include("means/mean.jl");
include("kernels/kernels.jl");
include("GP/GP.jl");
include("acquisition/acquisition.jl");
include("bayesopt/bayesopt.jl");

# export BayesOpt, ZeroMean, SquaredExponential

end
