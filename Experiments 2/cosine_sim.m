function sim = cosine_sim(vec1,vec2)
% calcualte cosine similarity

sim = (vec1*vec2)/(norm(vec1)*norm(vec2));

end