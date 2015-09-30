#    Copyright 2015 Raytheon BBN Technologies
#  
#     Licensed under the Apache License, Version 2.0 (the "License");
#     you may not use this file except in compliance with the License.
#     You may obtain a copy of the License at
#  
#       http://www.apache.org/licenses/LICENSE-2.0
#  
#     Unless required by applicable law or agreed to in writing, software
#     distributed under the License is distributed on an "AS IS" BASIS,
#     WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#     See the License for the specific language governing permissions and
#     limitations under the License.

module QuantumTomography

import Distributions.fit

export fit,
       predict,
       FreeLSStateTomo,
       LSStateTomo,
       MLStateTomo

using Convex, Distributions, SCS, QuantumInfo

function build_state_predictor(obs::Vector{Matrix})
    return reduce(vcat,[vec(o)' for o in obs])
end

function build_process_predictor(obs::Vector{Matrix}, prep::Vector{Matrix})
    exps = Matrix[ choi_liou_involution(vec(o)*vec(p)') for o in obs, p in prep ]
    return reduce(vcat, map(m->vec(m)', vec(exps)) )
end

"""
Free (unconstrained) least-squares state tomography algorithm. It is
constructed from a collection of observables corresponding to
measurements that are performed on the state being reconstructed.
"""
type FreeLSStateTomo
    inputdim::Int
    outputdim::Int
    pred::Matrix
    function LSStateTomo(obs::Vector)
        pred = build_state_predictor(obs)
        outputdim = size(pred,1)
        inputdim = size(pred,2)
        new(inputdim,outputdim,pred)
    end
end

"""
Predict outcomes of a tomography experiment for a given state (density matrix).
"""
function predict(method::FreeLSStateTomo, state)
    method.pred*vec(state)
end

"""
Reconstruct a state from observations (i.e., perform state tomography).
"""
function fit(method::FreeLSStateTomo, means::Vector{Float64}, vars::Vector{Float64}=-ones(length(means)); algorithm=:OLS)
    if length(means) != method.outputdim
        error("The number of expected means does not match the required number of experiments")
    end
    d = round(Int, method.inputdim |> sqrt)
    if method==:OLS
        reg = pred\means
        return reshape(reg,d,d), norm(pred*reg-means,2)/length(means), :Optimal
    elseif method==:GLS
        if any(vars<0)
            error("Variances must be positive for generalized least squares.")
        end
        reg = (sqrt(vars)\pred)\means
        return reshape((sqrt(vars)\pred)\means,d,d), sqrt(dot(pred*reg-means,diagm(vars)\(pred*reg-means)))/length(means), :Optimal
    else
        error("Unrecognized method for least squares state tomography")
    end
end


"""
Constrained least-squares state tomography algorithm. It is
constructed from a collection of observables corresponding to
measurements that are performed on the state being reconstructed.
"""
type LSStateTomo
    inputdim::Int
    outputdim::Int
    realpred::Matrix
    function LSStateTomo(obs::Vector)
        pred = build_state_predictor(obs)
        outputdim = size(pred,1)
        inputdim = size(pred,2)
        realpred = [real(pred) imag(pred)];
        new(inputdim,outputdim,realpred)
    end
end

function predict(method::LSStateTomo,state)
    (method.realpred[:,1:round(Int,end/2)]+1im*method.realpred[:,round(Int,end/2)+1:end])*vec(state)
end

function fit(method::LSStateTomo,
             means::Vector{Float64}, 
             vars::Vector{Float64};
             solver = SCSSolver(verbose=0, max_iters=10_000, eps = 1e-8))

    if length(means) != length(vars) || method.outputdim != length(means)
        error("Size of observations and/or predictons do not match.")
    end
    dsq = method.inputdim
    d = round(Int,sqrt(dsq))

    # We assume that the predictions are always real-valued
    # and we need to do the complex->real translation manually since
    # Convex.jl does not support complex numbers yet
    ivars = 1./sqrt(vars)

    ρr = Variable(d,d)
    ρi = Variable(d,d)

    constraints = trace(ρr) == 1
    constraints += trace(ρi) == 0
    constraints += isposdef([ρr ρi; -ρi ρr])

    # TODO: use quad_form instead of vecnorm? Have 1/vars are diagonal quadratic form
    problem = minimize( vecnorm( (means - method.realpred*[vec(ρr); vec(ρi)]) .* ivars, 2)^2, constraints )

    solve!(problem, solver)

    return (ρr.value - 1im*ρi.value), problem.optval, problem.status
end

"""
Maximum-likelihood quantum state tomography algorithm. It is
constructed from a collection of observables corresponding to
measurements that are performed on the state being reconstructed, as
well as a hedging factor β. If β=0, no hedging is applied. If β > 0
either a log determinant or log minimum eigenvalue penalty is applied.
"""
type MLStateTomo
    effects::Vector
    dim::Int64
    β::Float64
    function MLStateTomo(v::Vector,β=0.0)
        for e in v
            if !ishermitian(e) || !ispossemidef(e) || real(trace(e))>1
                error("MLStateTomo state tomography is parameterized by POVM effects only.")
            end
        end
        if !all([size(e,1)==size(e,2) for e in v]) || !all([size(v[1],1)==size(e,1) for e in v]) 
                error("All effects must be square matrices, and they must have have the same dimension.")
        end
        if β < 0
            error("Hedging penalty must be positive.")
        end
        new(v,size(v[1],1),β)
    end
end

function predict(method::MLStateTomo,ρ)
    Float64[ real(trace(ρ*e)) for e in method.effects]
end

function fit(method::MLStateTomo,
             counts::Vector{Int};
             solver = SCSSolver(verbose=0, max_iters=10_000, eps = 1e-8, warm_start=true))

    if length(method.effects) != length(counts)
        error("Vector of counts and vector of effects must have same length, but length(counts) == $(length(counts)) != $(length(method.effects))")
    end
    d = method.dim

    ρr = Variable(d,d)
    ρi = Variable(d,d)

    ϕ(m) = [real(m) -imag(m); imag(m) real(m)];
    ϕ(r,i) = [r i; -i r]

    obj = counts[1] * log(trace([ρr ρi; -ρi ρr]*ϕ(method.effects[1])))
    for i=2:length(method.effects)
        obj += counts[i] * log(trace([ρr ρi; -ρi ρr]*ϕ(method.effects[i])))
    end
    #obj += method.β!=0.0 ? method.β*logdet([ρr ρi; -ρi ρr]) : 0.0
    obj += method.β!=0.0 ? method.β*log(lambdamin([ρr ρi; -ρi ρr])) : 0.0

    constraints = trace(ρr) == 1
    constraints += isposdef([ρr ρi; -ρi ρr])

    problem = maximize(obj, constraints)

    solve!(problem, solver)

    return ρr.value-im*ρi.value, problem.optval, problem.status
end

function trb_sop(da,db)
    sop = spzeros(da^2,(da*db)^2)
    for i=0:da-1
        for j=0:da-1
            for k=0:db-1
                sop += vec(ketbra(i,j,da))*vec(kron(ketbra(i,j,da),ketbra(k,k,db)))'
            end
        end
    end
    return sop
end

# TODO: what about the (unobservable) traceful component
function qpt_lsq(pred::Matrix, means::Vector{Float64}, vars::Vector{Float64}; method=:OLS)
    d = Int(shape(pred,1) |> sqrt |> round)
    if method==:OLS
        return reshape(pred\means,d,d)
    elseif method==:GLS
        return reshape((sqrt(vars)\pred)\means,d,d)
    else
        error("Unrecognized method for least squares process tomography")
    end
end

type LSProcessTomo
    inputdim::Int
    outputdim::Int
    realpred::Vector{AbstractMatrix}
end

# For QPT, we write the predictor as operating on Choi-Jamilokoski
# matrices.  This is a bit awkward in comparisson to using the
# Liouville/natural representation, but it gets around some of the
# limitations of Convex.jl, and it is also much more efficient.
function qpt_ml(pred::Matrix, means::Vector{Float64}, vars::Vector{Float64})
    if length(means) != length(vars) || size(pred,1) != length(means)
        error("Size of observations and/or predictons do not match.")
    end
    d4 = size(pred,2)
    d2 = Int(sqrt(d4))
    d  = Int(sqrt(d2))

    # We assume that the predictions are always real-valued
    # and we need to do the complex->real translation manually since
    # Convex.jl does not support complex numbers yet
    rpred = [real(pred) imag(pred)];
    ivars = 1./sqrt(vars)

    ptrb = trb_sop(d,d)

    ρr = Variable(d2,d2)
    ρi = Variable(d2,d2)

    problem = minimize( vecnorm( (means - rpred*[vec(ρr); vec(ρi)]) .* ivars, 2 )^2 )

    problem.constraints += isposdef([ρr ρi; -ρi ρr])
    problem.constraints += trace(ρi) == 0
    problem.constraints += reshape(ptrb*vec(ρr),d,d) == eye(d)
    problem.constraints += reshape(ptrb*vec(ρi),d,d) == zeros(d,d)

    solve!(problem, SCSSolver(verbose=0))

    return (ρr.value - 1im*ρi.value), problem.optval, problem.status
end

function qst_ml(obs::Vector{Matrix}, means::Vector{Float64}, vars::Vector{Float64})
    #pred = reduce(vcat,[vec(o)' for o in obs])
    pred = build_state_predictor(obs)
    return qst_ml(pred, means, vars)
end

function qpt_ml(obs::Vector{Matrix}, states::Vector{Matrix}, means::Vector{Float64}, vars::Vector{Float64})
    #exps = Matrix[ choi_liou_involution(vec(o)*vec(p)') for o in obs, p in prep ]
    #pred = reduce(vcat, map(m->vec(m)', vec(exps)) )
    pred = build_process_predictor(obs, prep)
    return qpt_ml(pred, meas, vars)
end

include("utilities.jl")

end # module

