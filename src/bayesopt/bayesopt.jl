mutable struct BayesOpt
    gp::GP
    BayesOpt(X, Y, mean, kernel) = new(GP(X, Y, mean, kernel))
end