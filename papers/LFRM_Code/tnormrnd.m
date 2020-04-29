function [result] = tnormrnd(mu, st, lower, upper)
    
    one = normcdf(lower, mu, st);
    two = normcdf(upper, mu, st);
    three = unifrnd(one, two);
    result = norminv(three, mu, st);
    

% norminv(unifrnd(normcdf(lower_bound,mean,variance),normcdf(upper_bound,me
% an,variance)),mean, variance)