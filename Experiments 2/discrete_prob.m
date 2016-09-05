function [ sample ] = discrete_prob( probability )
%  returns a sample from a discrete distribution
%         Inputs
%         ------
%             probability: vector summing to 1
%         Outputs
%         -------
%             sample: integer
            
            
if abs(sum(probability) - 1.0) > 1e-6
    error('Not a valid probability distribution')
end

cdf = cumsum(probability);
samp = rand();
i = 1;

while(true)
    if samp <= cdf(i)
        sample = i;
        break;
    else
        i = i+1;
    end
end
end

