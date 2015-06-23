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

using Convex, Distributions, SchattenNorms, SCS

set_default_solver(SCSSolver(verbose=0,max_iters=10000,eps=1e-5))

function ketbra(a,b,d)
    m = spzeros(Float64,d,d)
    m[a+1,b+1] = 1.0
    return m
end

function qst_ml(pred::Matrix, means::Vector{Float64}, vars::Vector{Float64})
    if length(means) != length(vars) || size(pred,1) != length(means)
        error("Size of observations and/or predictons do not match.")
    end
    dsq = size(pred,2)
    d = Int(sqrt(dsq))
    # We assume that the predictions are always real-valued
    # and we need to do the complex->real translation manually since
    # Convex.jl does not support complex numbers yet
    rpred = [real(pred) imag(pred)];
    ivars = 1./sqrt(vars)

    ρr = Variable(d,d)
    ρi = Variable(d,d)

    problem = minimize( vecnorm( (means - rpred*[vec(ρr); vec(ρi)]) .* ivars, 2)^2 )

    problem.constraints += trace(ρr) == 1
    problem.constraints += trace(ρi) == 0
    problem.constraints += isposdef([ρr ρi; -ρi ρr])

    solve!(problem, SCSSolver(verbose=0))

    return (ρr.value - 1im*ρi.value), problem.optval, problem.status
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
    pred = reduce(vcat,[vec(o)' for o in obs])
    return qst_ml(pred, means, vars)
end

function qpt_ml(obs::Vector{Matrix}, states::Vector{Matrix}, means::Vector{Float64}, vars::Vector{Float64})
    exps = Matrix[ choi_liou_involution(vec(o)*vec(p)') for o in obs, p in prep ]
    pred = reduce(vcat, map(m->vec(m)', vec(exps)) )
    return qpt_ml(pred, meas, vars)
end

