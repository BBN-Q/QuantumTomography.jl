export FreeLSProcessTomo

function build_liou_process_predictor(prep::Vector, obs::Vector)
    exps = Matrix[ vec(p)*vec(o)' for o in obs, p in prep ]
    return reduce(vcat, map(m->vec(m)', vec(exps)) )
end

"""
`FreeLSProcessTomo(states::Vector,obs::Vector)`

Free (unconstrained) least-squares process tomography algorithm. It is
constructed from a vector of states that are used to probe a process, and
a vector of observables that are used to measure the probe states.
"""
type FreeLSProcessTomo
    statecount::Int
    obscount::Int
    inputdim::Int
    outputdim::Int
    pred::Matrix
    function FreeLSProcessTomo(states::Vector,obs::Vector)
        pred = build_liou_process_predictor(states, obs)
        outputdim = size(pred,1)
        inputdim = size(pred,2)
        new(length(states),length(obs),inputdim,outputdim,pred)
    end
end

function predict(method::FreeLSProcessTomo, process::Matrix)
    reshape(method.pred*vec(process), (method.statecount,method.obscount))
end

function fit(method::FreeLSProcessTomo,
             means::Vector,
             vars::Vector{Float64}=-ones(length(means));
             algorithm=:OLS)
    if length(means) != method.outputdim
        error("The number of expected means does not match the required number of experiments")
    end
    d = round(Int, size(method.pred,2) |> sqrt)
    if algorithm==:OLS
        reg = method.pred\means
        return (reshape(reg,d,d),
                norm(method.pred*reg-means,2)^2,
                :Optimal)
    elseif algorithm==:GLS
        if any(vars<0)
            error("Variances must be positive for generalized least squares process tomography.")
        end
        return (reshape((sqrt(vars)\method.pred)\means,d,d),
                norm((method.pred*reg-means)./sqrt(vars),2)^2,
                :Optimal)
    else
        error("Unrecognized method for least squares process tomography")
    end
end

function fit(method::FreeLSProcessTomo,
             means::Matrix,
             vars::Matrix{Float64}=-ones(size(means));
             algorithm=:OLS)
    if size(means) != (method.statecount, method.obscount)
        error("Mean matrix must be $((method.statecount, methods.obscount)), but is $(size(means))")
    elseif algorithm==:GLS && size(vars) != (method.statecount, methods.obscount)
        error("Pointwise variance matrix must be $((method.statecount, methods.obscount)), but is $(size(means))")
    end
    fit(method, vec(means), vec(vars), algorithm=algorithm)
end

type LSProcessTomo
    inputdim::Int
    outputdim::Int
    realpred::Vector{AbstractMatrix}
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
# matrices.  This is a bit awkward in comparison to using the
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
