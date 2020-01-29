cd(@__DIR__)
using Pkg
pkg"activate ."
using Soss, MonteCarloMeasurements, Distributions
nr = 5 # Number of reviewers
na = 20 # Number of articles
reviewer_bias = rand(Normal(0,1), nr)
article_score = rand(Normal(0,2), na)
Rtrue = clamp.([r+a for r in reviewer_bias, a in article_score], -5, 5)
R = Rtrue .+ 0.5 .* randn.()
Rmask = rand(Bool, size(R))
R = replace(Rmask, 0=>missing) .* R
Rv = [R[i,j] for i in axes(R,1), j in axes(R,2) if !ismissing(R[i,j])]
indv = [(i,j) for i in axes(R,1), j in axes(R,2) if !ismissing(R[i,j])]
Rv .-= mean(Rv)

# m = @model begin
#     article_pop_variance ~ Normal(2,0.1)
#     reviewer_bias ~ Normal(0, 1) |> iid(nr)
#     reviewer_gain ~ Normal(1, 0.1) |> iid(nr)
#     true_article_score ~ Normal(0,article_pop_variance) |> iid(na)
#     rσ ~ Gamma(0.2)
#     R ~ For(nr, na) do i,j
#         Normal(reviewer_bias[i] + true_article_score[j] + reviewer_gain[i]*true_article_score[j], rσ)
#     end
# end;

tm = @model begin
    rσ ~ Gamma(0.2)
    article_pop_variance ~ TruncatedNormal(1.,0.1, 0, 100)
    reviewer_bias ~ Normal(0, 1) |> iid(nr)
    reviewer_noise ~ TruncatedNormal(rσ, 0.1,0,3) |> iid(nr) # Different reviewer have different noise variances
    reviewer_gain ~ Normal(1, 0.1) |> iid(nr)
    true_article_score ~ Normal(0,article_pop_variance) |> iid(na)
    Rv ~ For(length(indv)) do ind
        i,j = indv[ind]
        Normal(reviewer_bias[i] + true_article_score[j] + reviewer_gain[i]*true_article_score[j], reviewer_noise[i])
    end
end;


# s = rand(m(R=R))
# norm(s.R - R)
s = rand(tm(Rv=Rv))
truth = rand(tm())
norm(s.Rv - Rv)

observed_score = map(1:na) do i
    mean([truth.Rv[ind] for ind in eachindex(indv) if indv[ind][2] == i])
end


@time post = dynamicHMC(tm(), (Rv=truth.Rv,), 2000);
p = particles(post[300:end])


##
# p = particles([rand(tm(Rv=Rv)) for _ in 1:1])
figs = map((:reviewer_bias, :reviewer_gain, :true_article_score)) do s
    bar(getproperty(truth, s))
    prop = getproperty(p, s)
    errorbarplot!(1:length(prop), prop, seriestype=:scatter, legend=false, title=string(s), m=2)
end
plot(figs...)


##
@info "Percentage of correct rank from model $(mean(sortperm(truth.true_article_score) .==  sortperm(mean.(p.true_article_score))))"
@info "Percentage of correct rank from observed score $(mean(sortperm(truth.true_article_score) .==  sortperm(observed_score)))"
scatter([(truth.true_article_score) (observed_score)], label=["true" "obs"])
errorbarplot!(1:na, p.true_article_score, seriestype=:scatter, lab="model", m=(2,))
