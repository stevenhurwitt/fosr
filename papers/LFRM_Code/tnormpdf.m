function [prob] = tnormpdf(x, mu, st, lower, upper)

    prob = log(normpdf(x, mu, st)) - log(normcdf(upper, mu, st) - normcdf(lower, mu, st));
    