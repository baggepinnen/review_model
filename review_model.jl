# This is an experiment on using probabilistic programming (using Soss.jl and Turing.jl) to calibrate review scores given by `na` reviewers to `na` articles submitted to, e.g., a conference for publication. Each reviewer assigns each article a score between 1 and 5. We assume there are no missing reviews for simplicity. The review scores are stored in the vector `Rv`. The reviewers might not use a "metric scale" for their scores, i.e., difference between score 3-4 might be larger or smaller than the difference between 9-10. Du to this, we make use of ordinal regression, where we estimate the scale the reviewers are using. In reality, we might want to have a separate scsale for each reviewer, but the number of reviews per reviewer reqiured to estimate this accurately would be too high, so we settle for one single scale for all of them.
cd(@__DIR__)
using Pkg
pkg"activate ."
using Soss, MonteCarloMeasurements, Distributions, Plots, LinearAlgebra, Statistics, Turing, NamedTupleTools
default(size=(500,300))
nr = 10  # Number of reviewers
na = 15 # Number of articles
indv = [(i,j) for i in 1:nr, j in 1:na]
max_score = 5
min_score = 1
nscores = max_score-min_score+1;

# Below, we define some helper functions
function Distributions.cdf(R::AbstractArray,mi=minimum(skipmissing(R)),ma=maximum(skipmissing(R)))
    range = mi:ma
    Ro = copy(R)
    cumdist = map(range) do level
        count(R .<= level) / length(R)
    end
    cumdist
end

function rankdist(x,y)
    d = 0
    for xi in eachindex(x)
        yi = findfirst(==(x[xi]), y)
        d += abs(xi-yi)
    end
    d/length(x)
end

function logodds(R::AbstractArray)
    cumdist = cdf(R)
    mi,ma = extrema(R)
    range = mi:ma
    Ro = copy(R)
    map(R) do r
        ind = findfirst(==(r), range)
        logit(cumdist[ind])
    end
end;

# This function accepts a vector of differences between cutpoints and form the cumulative sum in both directions to form the final cutpoints. The reason for calculating the cumulative sum in both directions is that the variance grows as you add uncertain variables, and with this approach, the variance will be highest in the middle, which agrees with my intuition on how flexible the cutpoints should be.
cumcut(diffcp) = ((cumsum(diffcp) + reverse(1  .- cumsum(reverse(diffcp)))) ./ 2)*max_score .- nscores/2;

# # Soss
# First out is the Soss model
cum_model = Soss.@model indv begin
    article_pop_std ~ truncated(Normal(0.5, 0.1), 0, 100) # The population of all article scores has a common variance
    reviewer_gain ~ truncated(Normal(1, 0.5), 0.1, 3) |> iid(nr) # Each reviewer has a unique gain, i.e., when an article gets better or worse, the reviewer adjusts the score more or less.
    article_score ~ Normal(0,article_pop_std) |> iid(na) # The true, calibrated article score
    diffcp ~ Dirichlet(nscores-1,50) # Vector of differences between cutpoints
    cutpoints = cumcut(diffcp)
    # μ = (max_score-min_score)/2 + 1

    Rv ~ For(length(indv)) do ind
        i,j = indv[ind]
        pred = article_score[j] + reviewer_gain[i]*article_score[j] # linear model predicting the log-odds of the review score, this will be roughly between -4 and 4
        OrderedLogistic(pred,cutpoints) # The observed review score is an ordered logistic variable. This transform the `pred` to a categorical value between 1 and 10
    end
end;

# We sample some points from the prior and see what it thinks about the world of reviews
prior_sample = rand(cum_model(indv=indv), 1000) # rand(::SossModel) does not seem to accept a number of samples
prior_sample = delete.(prior_sample, :indv) |> particles
errorbarplot(prior_sample.Rv, title="Review score prior") |> display
# We can visualize what the prior considers possible cutpoints for the log-odds
mcplot(prior_sample.cutpoints, title="Cutpoints prior") |> display
# We can also visualize the prior distribution over observed review scores and log-odds
histogram(reduce(union,prior_sample.Rv), title="Samples of review scores from prior", xlabel="Review score")
#


# below, we sample one data point from the model and call this the "truth".
truth = rand(cum_model(indv=indv));

# We now perform inference on the model using as observed variables the review scores from the "truth"
@time post = dynamicHMC(cum_model(indv=indv), (Rv=truth.Rv,), 4200);
# This call takes a long time. It seems to be taking long time before it even starts to sample.
p = particles(post[201:end]); # Convert posterior to `Particles`

# We now visualize the inferred articles scores and reviewer gains and compare the posterior to what we know are the generating parameters from the `truth`
figs = map((:reviewer_gain, :article_score)) do s
    bar(getproperty(truth, s))
    prop = getproperty(p, s)
    errorbarplot!(prop, seriestype=:scatter, legend=false, title=string(s), m=2)
end
plot(figs...)
# using Soss, there is a clear indication that the model has learned at least something about the reviewer gains, even though the variance is high.

# We can calculate the log-odds of the observed scores for each article and compare this to the models predictions
# lo = logodds(truth.Rv)
observed_score = map(1:na) do j
    mean([truth.Rv[ind] for ind in eachindex(indv) if indv[ind][2] == j])
end
observed_score .-= mean(observed_score)

function print_rank_results(truth, p, observed_score)

    println("Percentage of correct rank from model $(mean(sortperm(truth.article_score) .==  sortperm(mean.(p.article_score))))")
    println("Percentage of correct rank from observed score $(mean(sortperm(truth.article_score) .==  sortperm(observed_score)))")

    println("Percentage of top 50% correct from model $(mean(sortperm(truth.article_score)[1:end÷2] .∈  Ref(sortperm(mean.(p.article_score))[1:end÷2])))")
    println("Percentage of top 50% correct from observed score $(mean(sortperm(truth.article_score)[1:end÷2] .∈  Ref(sortperm(observed_score)[1:end÷2])))")

    println("Rankdist between correct rank and model $(rankdist(sortperm(truth.article_score),  sortperm(mean.(p.article_score))))") # `rankdist` is a distance measure I cooked up to try to measure the permutation distance between two rank vectors.
    println("Rankdist between correct rank and observed score $(rankdist(sortperm(truth.article_score),  sortperm(observed_score)))")
    scatter([(truth.article_score) (observed_score)], label=["true" "obs"], title="Article scores", xlabel="Article number")
    errorbarplot!(1:na, p.article_score, seriestype=:scatter, lab="model", m=(2,))
end
print_rank_results(truth, p, observed_score)

# # Turing
# We now perform the same exercise with Turing, the model should be exactly the same with the same numerical values for the parameters

using Turing, Turing2MonteCarloMeasurements, NamedTupleTools
Turing.@model cum_model(indv, Rv, ::Type{T}=Float64) where {T} = begin
    article_pop_std ~ truncated(Normal(1., 0.1), 0, 100)
    reviewer_noise     = Vector{T}(undef, nr)
    pred               = Vector{T}(undef, length(indv))

    reviewer_gain ~ MvNormal(fill(1,nr), 0.15^2)
    article_score ~ MvNormal(zeros(na),article_pop_std^2)

    diffcp ~ Dirichlet(nscores-1,50)
    cutpoints = cumcut(diffcp)
    for ind in eachindex(indv)
        i,j = indv[ind]
        pred[ind] = article_score[j] + reviewer_gain[i]*article_score[j]
        Rv[ind] ~ OrderedLogistic(pred[ind],cutpoints)
    end
    @namedtuple(Rv, article_score, cutpoints, reviewer_noise, reviewer_gain, pred, diffcp, z, rσ, article_pop_std)
end;

# Once again we sample one data points and call it the truth
prior = cum_model(indv, Union{Int,Missing}[fill(missing, length(indv))...])
## truth = prior() # We use the truth from Soss
prior_sample = [prior() for _ in 1:500] |> particles
errorbarplot(1:length(indv), prior_sample.Rv, 0.0, title="Review score prior") |> display
# We can visualize what the prior considers possible cutpoints for the log-odds
mcplot(1:length(prior_sample.cutpoints), prior_sample.cutpoints, title="Cutpoints prior") |> display
# We can also visualize the prior distribution over observed review scores and log-odds
histogram(reduce(union,prior_sample.Rv), title="Samples of review scores from prior", xlabel="Review score")
#
histogram(reduce(union,prior_sample.pred), title="Samples of log-odds predictions from prior")
# We now sample from the posterior using Turing
m = cum_model(indv, Int.(truth.Rv))
@time chain = sample(m, HMC(0.03, 7), 5000) # NUTS does not work
p = Particles(chain, crop=1000);
# Turing samples *much* faster that Soss for this model
dc = describe(chain, digits=3, q=[0.1, 0.5, 0.9])
#
dc[1].df.r_hat
# Maximum r_hat
maximum(filter(isfinite, dc[1].df.r_hat))
#
median(filter(isfinite, dc[1].df.r_hat))

# We plot the same figure of the posterior article scores and reviewer gains as we did for Soss
figs = map((:article_score,:reviewer_gain)) do s
    bar(getproperty(truth, s))
    prop = getproperty(p, s)
    errorbarplot!(1:length(prop), prop, seriestype=:scatter, legend=false, title=string(s), m=2)
end
plot(figs...)
# *When sampling with turing, the posterior for reviewer gain is very different from when sampling with Soss*

# We also check how the model fared when estimating the rank of the articles
lo = logodds(truth.Rv)
observed_score = map(1:na) do j
    median([lo[ind] for ind in eachindex(indv) if indv[ind][2] == j])
end
print_rank_results(truth, p, observed_score)


#src literateweave("review_model.jl", doctype="md2pdf", latex_cmd="lualatex --output-directory=build", template="template.tpl")
