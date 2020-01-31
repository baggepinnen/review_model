cd(@__DIR__)
using Pkg
pkg"activate ."
using Soss, MonteCarloMeasurements, Distributions, Turing
nr = 5 # Number of reviewers
na = 20 # Number of articles
max_score = 10
reviewer_bias = rand(Normal(0,1), nr)
article_score = rand(Normal(0,2), na)
Rtrue = clamp.([r+a for r in reviewer_bias, a in article_score], -5, 5)
R = Rtrue .+ 0.5 .* randn.()
R = round.(Int,R)
Rmask = rand(Binomial(1,0.7), size(R))
R = replace(Rmask, 0=>missing) .* R
Rv = [R[i,j] for i in axes(R,1), j in axes(R,2) if !ismissing(R[i,j])]
indv = [(i,j) for i in axes(R,1), j in axes(R,2) if !ismissing(R[i,j])]
# Rv .-= mean(Rv)
min_score = -5
max_score = 5
nscores = max_score-min_score+1


function Distributions.cdf(R::AbstractArray,mi=minimum(skipmissing(R)),ma=maximum(skipmissing(R)))
    range = mi:ma
    Ro = copy(R)
    cumdist = map(range) do level
        count(R .<= level) / length(R)
    end
    cumdist
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
end



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

# standard_model = @model indv begin
#     rσ ~ Gamma(0.2)
#     article_pop_variance ~ TruncatedNormal(1.,0.1, 0, 100)
#     reviewer_bias ~ Normal(0, 1) |> iid(nr)
#     reviewer_noise ~ TruncatedNormal(rσ, 0.1,0,3) |> iid(nr) # Different reviewer have different noise variances
#     reviewer_gain ~ Normal(1, 0.1) |> iid(nr)
#     true_article_score ~ Normal(0,article_pop_variance) |> iid(na)
#     Rv ~ For(length(indv)) do ind
#         i,j = indv[ind]
#         Normal(reviewer_bias[i] + true_article_score[j] + reviewer_gain[i]*true_article_score[j], reviewer_noise[i])
#     end
# end;
#
# s = rand(standard_model(indv=indv))
# truth = rand(standard_model(indv=indv))


cum_model = Soss.@model indv begin
    rσ ~ Gamma(0.2)
    article_pop_variance ~ truncated(Normal(1., 0.5), 0, 100)
    reviewer_noise ~ truncated(Normal(rσ, 0.1), 0, 3) |> iid(nr) # Different reviewer have different noise variances
    reviewer_gain ~ Normal(1, 0.1) |> iid(nr)
    true_article_score ~ Normal(0,article_pop_variance) |> iid(na)
    diffcp ~ For(nscores-1) do i
        # i == 1 ? truncated(Normal(1,0.3), 0, 3) : Normal(1,0.15)
        Normal(1,0.15)
    end
    #cumsum(diffcp) .- (abs(min_score) + 1)
    pred ~ For(length(indv)) do ind
        i,j = indv[ind]
        Normal(true_article_score[j] + reviewer_gain[i]*true_article_score[j], reviewer_noise[i])
    end
    Rv ~ For(length(indv)) do ind
        i,j = indv[ind]
        Normal(pred[ind], 1)#OrderedLogistic(pred[ind],cutpoints)
    end
end;

# s = rand(m(R=R))
# norm(s.R - R)

s = [rand(cum_model(indv=indv)) for _ in 1:1000]
truth = rand(cum_model(indv=indv))


observed_score = map(1:na) do j
    mean([truth.Rv[ind] for ind in eachindex(indv) if indv[ind][2] == j])
end


@time post = dynamicHMC(cum_model(indv=indv), (Rv=truth.Rv,), 1000);
@show maximum([logpdf(cum_model(Rv=Rv), p) for p in post])
p = particles(post[200:end])


##
# p = particles([rand(standard_model(Rv=Rv)) for _ in 1:1])
figs = map((:reviewer_gain, :true_article_score)) do s
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

##
pp = map(1:length(indv)) do ind
    i,j = indv[ind]
    p.true_article_score[j] + p.reviewer_gain[i]*p.true_article_score[j]
end
scatter(lq)
errorbarplot!(1:length(pp), pp, seriestype=:scatter)
##








using Turing

# Base.delete_method.(methods(logodds))


model = Soss.@model begin
    # q ~ For(na) do j
    #     For(nr) do i
    #         Soss.LogitNormal(0,1)
    #     end
    # end
    diffcp ~ For(max_score-1) do i
                Normal(1,0.15)
             end
    cutpoints = cumsum(diffcp)
    pred ~ For(na) do j
            For(nr) do i
                Categorical(max_score)
            end
    end
    R ~ For(na) do j
            For(nr) do i
                OrderedLogistic(pred[j][i],cutpoints)
            end
    end

end;

lq = logodds(Rv)

truth = rand(model())
s = rand(model(R = R))

@time post = dynamicHMC(model(), (pred=truth.pred,), 1000);
@time post = dynamicHMC(model(), (R=truth.R,), 1000);

c = cdf.(Rc,0,10)
p = diff.(c)
sum.(p)


## Turing cumulative model

using Turing2MonteCarloMeasurements
Turing.@model cum_model(indv, Rv) = begin
    rσ ~ Gamma(0.2)
    article_pop_variance ~ truncated(Normal(1., 0.5), 0, 100)
    reviewer_noise     = Vector{Real}(undef, nr)
    reviewer_gain      = Vector{Real}(undef, nr)
    true_article_score = Vector{Real}(undef, na)
    z                  = Vector{Real}(undef, length(indv))
    pred               = Vector{Real}(undef, length(indv))
    # Rv                 = Vector{Real}(undef, length(indv))
    diffcp             = Vector{Real}(undef, nscores-1)
    offset             = (abs(min_score)) + 1

    for i = 1:nr
        reviewer_noise[i] ~ truncated(Normal(rσ, 0.1), 0, 3)
        reviewer_gain[i] ~ Normal(1, 0.1)
    end
    for j = 1:na
        true_article_score[j] ~ Normal(0,article_pop_variance)
    end
    for i = 1:nscores-1
        # i == 1 ? truncated(Normal(1,0.3), 0, 3) : Normal(1,0.15)
        diffcp[i] ~ Normal(1,0.15)
    end
    cutpoints = cumsum(diffcp)
    for ind in eachindex(indv)
        i,j = indv[ind]
        z[ind] ~ Normal(0,1)
        pred[ind] = true_article_score[j] + reviewer_gain[i]*true_article_score[j] + reviewer_noise[i]*z[ind] + offset
        Rv[ind] ~ OrderedLogistic(pred[ind],cutpoints)
    end
end;

m = cum_model(indv, Rv .+ (abs(min_score)+1))

chain = sample(m, HMC(0.03, 10), 2000)
p = Particles(chain, crop=500)
describe(chain)

figs = map((:true_article_score,)) do s
    bar(@eval($s))
    prop = getproperty(p, s)
    errorbarplot!(1:length(prop), prop, seriestype=:scatter, legend=false, title=string(s), m=2)
end
plot(figs...)

truth = (true_article_score=article_score, reviewer_bias=reviewer_bias, Rv=Rv)

observed_score = map(1:na) do j
    mean([truth.Rv[ind] for ind in eachindex(indv) if indv[ind][2] == j])
end


@info "Percentage of correct rank from model $(mean(sortperm(truth.true_article_score) .==  sortperm(mean.(p.true_article_score))))"
@info "Percentage of correct rank from observed score $(mean(sortperm(truth.true_article_score) .==  sortperm(observed_score)))"
scatter([(truth.true_article_score) (observed_score)], label=["true" "obs"])
errorbarplot!(1:na, p.true_article_score, seriestype=:scatter, lab="model", m=(2,))

##
pp = map(1:length(indv)) do ind
    i,j = indv[ind]
    p.true_article_score[j] + p.reviewer_gain[i]*p.true_article_score[j]
end
scatter(lq)
errorbarplot!(1:length(pp), pp, seriestype=:scatter)
