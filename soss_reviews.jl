cd(@__DIR__)
using Pkg
pkg"activate ."
using Soss, MonteCarloMeasurements, Distributions, Turing
nr = 10 # Number of reviewers
na = 50 # Number of articles
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
# shuffle!(indv)
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
end



# m = @model begin
#     article_pop_variance ~ Normal(2,0.1)
#     reviewer_bias ~ Normal(0, 1) |> iid(nr)
#     reviewer_gain ~ Normal(1, 0.1) |> iid(nr)
#     article_score ~ Normal(0,article_pop_variance) |> iid(na)
#     rσ ~ Gamma(0.2)
#     R ~ For(nr, na) do i,j
#         Normal(reviewer_bias[i] + article_score[j] + reviewer_gain[i]*article_score[j], rσ)
#     end
# end;

# standard_model = @model indv begin
#     rσ ~ Gamma(0.2)
#     article_pop_variance ~ TruncatedNormal(1.,0.1, 0, 100)
#     reviewer_bias ~ Normal(0, 1) |> iid(nr)
#     reviewer_noise ~ TruncatedNormal(rσ, 0.1,0,3) |> iid(nr) # Different reviewer have different noise variances
#     reviewer_gain ~ Normal(1, 0.1) |> iid(nr)
#     article_score ~ Normal(0,article_pop_variance) |> iid(na)
#     Rv ~ For(length(indv)) do ind
#         i,j = indv[ind]
#         Normal(reviewer_bias[i] + article_score[j] + reviewer_gain[i]*article_score[j], reviewer_noise[i])
#     end
# end;
#
# s = rand(standard_model(indv=indv))
# truth = rand(standard_model(indv=indv))

cumcut(diffcp) = ((cumsum(diffcp) + reverse(1  .- cumsum(reverse(diffcp)))) ./ 2)*12 .- 6


cum_model = Soss.@model indv begin
    rσ ~ Gamma(0.2)
    article_pop_variance ~ truncated(Normal(1., 0.2), 0, 100)
    reviewer_noise ~ truncated(Normal(rσ, 0.1), 0, 3) |> iid(nr) # Different reviewer have different noise variances
    reviewer_gain ~ Normal(1, 0.1) |> iid(nr)
    article_score ~ Normal(0,article_pop_variance) |> iid(na)
    diffcp ~ Dirichlet(nscores-1,20)
    cutpoints = cumcut(diffcp)

    z ~ Normal(0,1) |> iid(length(indv))
    Rv ~ For(length(indv)) do ind
        i,j = indv[ind]
        pred = article_score[j] + reviewer_noise[i]*z[ind] + reviewer_gain[i]*article_score[j]
        OrderedLogistic(pred,cutpoints)
    end
end;

# s = rand(m(R=R))
# norm(s.R - R)

s = [rand(cum_model(indv=indv)) for _ in 1:1000]
truth = rand(cum_model(indv=indv))




@time post = dynamicHMC(cum_model(indv=indv), (Rv=truth.Rv,), 1000);
# @show maximum([logpdf(cum_model(Rv=Rv), p) for p in post])
p = particles(post[200:end])


##
# p = particles([rand(standard_model(Rv=Rv)) for _ in 1:1])
figs = map((:reviewer_gain, :article_score)) do s
    bar(getproperty(truth, s))
    prop = getproperty(p, s)
    errorbarplot!(1:length(prop), prop, seriestype=:scatter, legend=false, title=string(s), m=2)
end
plot(figs...)


##
lo = logodds(truth.Rv)
observed_score = map(1:na) do j
    median([lo[ind] for ind in eachindex(indv) if indv[ind][2] == j])
end


@info "Percentage of correct rank from model $(mean(sortperm(truth.article_score) .==  sortperm(mean.(p.article_score))))"
@info "Percentage of correct rank from observed score $(mean(sortperm(truth.article_score) .==  sortperm(observed_score)))"

@info "Rankdist between correct rank and model $(rankdist(sortperm(truth.article_score),  sortperm(mean.(p.article_score))))"
@info "Rankdist between correct rank and observed score $(rankdist(sortperm(truth.article_score),  sortperm(observed_score)))"
scatter([(truth.article_score) (observed_score)], label=["true" "obs"])
errorbarplot!(1:na, p.article_score, seriestype=:scatter, lab="model", m=(2,))


##
pp = map(1:length(indv)) do ind
    i,j = indv[ind]
    p.article_score[j] + p.reviewer_gain[i]*p.article_score[j]
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

using Turing2MonteCarloMeasurements, NamedTupleTools
Turing.@model cum_model(indv, Rv, ::Type{T}=Float64) where {T} = begin
    rσ ~ Gamma(0.2)
    article_pop_variance ~ truncated(Normal(0.9^2, 0.1), 0, 1.5)
    reviewer_noise     = Vector{T}(undef, nr)
    # reviewer_gain      = Vector{T}(undef, nr)
    # article_score = Vector{T}(undef, na)
    # z                  = Vector{T}(undef, length(indv))
    # Rv                 = Vector{T}(undef, length(indv))
    # diffcp             = Vector{T}(undef, nscores-2)
    pred               = Vector{T}(undef, length(indv))

    for i = 1:nr
        reviewer_noise[i] ~ truncated(Normal(rσ, 0.01), 0, 3)
    end
    reviewer_gain ~ MvNormal(fill(1,nr), 0.15^2)
    article_score ~ MvNormal(zeros(na),article_pop_variance)

    # for i = 1:length(diffcp)
    #     diffcp[i] ~ Normal(0,0.1)
    # end
    # cutpoints = similar(diffcp, nscores-1)
    # cutpoints[1] ~ Normal(-2,2)
    # for i = 2:length(cutpoints)
    #     cutpoints[i] = cutpoints[i-1] + exp(diffcp[i-1])
    # end
    # α ~ Normal(20,5)
    diffcp ~ Dirichlet(nscores-1,2000)
    cutpoints = ((cumsum(diffcp) + reverse(1  .- cumsum(reverse(diffcp)))) ./ 2)*12 .- 6#[2:end-1]
    # cutpoints = (cumsum(diffcp)*10 .- 5)#[2:end-1]
    # cutpoints = collect(1:10)
    z ~ MvNormal(zeros(length(indv)),1)
    for ind in eachindex(indv)
        i,j = indv[ind]
        pred[ind] = article_score[j] + reviewer_noise[i]*z[ind] + reviewer_gain[i]*article_score[j]
        Rv[ind] ~ OrderedLogistic(pred[ind],cutpoints)
    end
    @namedtuple(Rv, article_score, cutpoints, reviewer_noise, reviewer_gain, pred, diffcp, z, rσ, article_pop_variance)
end;



prior = cum_model(indv, Union{Int,Missing}[fill(missing, length(indv))...])
truth = prior()
prior_sample = [prior() for _ in 1:500] |> particles
errorbarplot(1:length(indv), prior_sample.Rv, 0.0) |> display
# errorbarplot(1:length(prior_sample.cutpoints), prior_sample.cutpoints, 0) |> display
mcplot(1:length(prior_sample.cutpoints), prior_sample.cutpoints) |> display

histogram(reduce(union,prior_sample.Rv))
histogram(reduce(union,prior_sample.pred))
##
m = cum_model(indv, Int.(truth.Rv))

chain = sample(m, HMC(0.03, 7), 1200)
# chain = sample(m, PG(10), 2000)
# chain = sample(m, NUTS(), 150)
p = Particles(chain, crop=200)
# describe(chain)

figs = map((:article_score,:reviewer_gain)) do s
    bar(getproperty(truth, s))
    prop = getproperty(p, s)
    errorbarplot!(1:length(prop), prop, seriestype=:scatter, legend=false, title=string(s), m=2)
end
plot(figs...)

i,j = indv[50]
r1 = [p.reviewer_gain[i], p.reviewer_noise[i], p.article_score[j], p.z[50]]

# truth = (article_score=article_score, reviewer_bias=reviewer_bias, Rv=Rv)

lo = logodds(truth.Rv)
observed_score = map(1:na) do j
    median([lo[ind] for ind in eachindex(indv) if indv[ind][2] == j])
end


@info "Percentage of correct rank from model $(mean(sortperm(truth.article_score) .==  sortperm(mean.(p.article_score))))"
@info "Percentage of correct rank from observed score $(mean(sortperm(truth.article_score) .==  sortperm(observed_score)))"

@info "Rankdist between correct rank and model $(rankdist(sortperm(truth.article_score),  sortperm(mean.(p.article_score))))"
@info "Rankdist between correct rank and observed score $(rankdist(sortperm(truth.article_score),  sortperm(observed_score)))"
scatter([(truth.article_score) (observed_score)], label=["true" "obs"])
errorbarplot!(1:na, p.article_score, seriestype=:scatter, lab="model", m=(2,))

##
pp = map(1:length(indv)) do ind
    i,j = indv[ind]

    pred = p.article_score[j] + p.reviewer_gain[i]*p.article_score[j] + p.reviewer_noise[i]*p.z[ind]
end
bar(truth.pred)
errorbarplot!(1:length(pp), pp, seriestype=:scatter)

using MCMCChains, StatsPlots
plot(chain)



@bymap rankdist(sortperm(truth.article_score),  @bymap(sortperm((p.article_score))))


## MAP

function get_nlogp(model)
    # Construct a trace struct
    vi = Turing.VarInfo(model)

    # Define a function to optimize.
    function nlogp(sm)
        spl = Turing.SampleFromPrior()
        new_vi = Turing.VarInfo(vi, spl, sm)
        try
            model(new_vi, spl)
        catch
            return 1e4
        end
        -new_vi.logp
    end

    return nlogp
end

# Define our data points.
model = cum_model(indv, truth.Rv)
nlogp = get_nlogp(model)

using Optim

p0 = [truth.rσ;
    truth.article_pop_variance;
    truth.reviewer_noise;
    truth.reviewer_gain;
    truth.article_score;
    truth.diffcp;
    truth.z]

nlogp(p0)
result = Optim.optimize(nlogp, p0, GradientDescent(), Optim.Options(store_trace=true, show_trace=true, show_every=1, iterations=1000, allow_f_increases=false, time_limit=300, x_tol=0, f_tol=0, g_tol=1e-8, f_calls_limit=0, g_calls_limit=0), autodiff=:forward)


using ForwardDiff
H = ForwardDiff.hessian(nlogp, result.minimizer)
g = ForwardDiff.gradient(nlogp, result.minimizer)

cond(H)
