export FreeLSProcessTomo,
       LSProcessTomo

function build_liou_process_predictor(prep::Vector, obs::Vector)
    exps = [ vec(p)*vec(o)' for o in obs, p in prep ]
    return reduce(vcat, map(m->vec(m)', vec(exps)) )
end

function build_choi_process_predictor(prep::Vector, obs::Vector)
    exps = [ QuantumInfo.choi_liou_involution(vec(o)*vec(p)') for o in obs, p in prep ]
    return reduce(vcat, map(m->vec(m)', vec(exps)) )
end

function isstate(m::Matrix)
    LinearAlgebra.ishermitian(m) && isapprox(LinearAlgebra.tr(m),1.0)
end

"""
`FreeLSProcessTomo(states::Vector,obs::Vector)`

Free (unconstrained) least-squares process tomography algorithm. It is
constructed from a vector of states that are used to probe a process, and
a vector of observables that are used to measure the probe states.
"""
struct FreeLSProcessTomo
    statecount::Int
    obscount::Int
    inputdim::Int
    outputdim::Int
    pred::Matrix{ComplexF64}
    function FreeLSProcessTomo(states::Vector,obs::Vector)
        @assert all([LinearAlgebra.ishermitian(o) for o in obs]) "Observables must be Hermitian"
        @assert all([isstate(o) for o in states])  "States must be valid density matrices"

        pred = build_liou_process_predictor(states, obs)
        outputdim = size(pred,1)
        inputdim = size(pred,2)
        new(length(states),length(obs),inputdim,outputdim,pred)
    end
end

function predict(method::FreeLSProcessTomo, process::Matrix)
    reshape(real(method.pred*vec(process)), (method.statecount,method.obscount))
end

function fit(method::FreeLSProcessTomo,
             means::Vector)
    if length(means) != method.outputdim
        error("The number of sample means does not match the required number of experiments")
    end
    d = round(Int, size(method.pred, 2) |> sqrt)
    reg = method.pred \ means
    return (reshape(reg,d,d),
            LinearAlgebra.norm(method.pred*reg-means,2)^2,
            :Optimal)
end

function fit(method::FreeLSProcessTomo,
             means::Vector,
             vars::Vector{Float64})
    if length(means) != method.outputdim
        error("The number of sample means does not match the required number of experiments")
    end
    if length(vars) != method.outputdim
        error("The number of sample variances does not match the required number of experiments")
    end
    if any(vars.<0)
        error("Samples variances must be positive for generalized least squares process tomography.")
    end
    d = round(Int, size(method.pred,2) |> sqrt)
    # weighted least squares
    reg = (LinearAlgebra.Diagonal(1 ./ sqrt.(vars)) * method.pred) \ (LinearAlgebra.Diagonal(1 ./ sqrt.(vars)) * means)
    return (reshape(reg,d,d),
            LinearAlgebra.norm((method.pred*reg-means) ./ sqrt.(vars),2)^2,
            :Optimal)
end

function fit(method::FreeLSProcessTomo,
             means::Matrix)
    if size(means) != (method.statecount, method.obscount)
        error("Mean matrix must be $((method.statecount, method.obscount)), but is $(size(means))")
    end
    fit(method, vec(means))
end

function fit(method::FreeLSProcessTomo,
             means::Matrix,
             vars::Matrix{Float64})
    if size(means) != (method.statecount, method.obscount)
        error("Mean matrix must be $((method.statecount, method.obscount)), but is $(size(means))")
    elseif size(vars) != (method.statecount, method.obscount)
        error("Pointwise variance matrix must be $((method.statecount, method.obscount)), but is $(size(means))")
    end
    fit(method, vec(means), vec(vars))
end

"""
`LSProcessTomo(states::Vector,obs::Vector)`

Least-squares process tomography algorithm where the result is
contrained to be completely positive (CP) and trace perserving
(TP). It is constructed from a vector of states that are used to probe
a process, and a vector of observables that are used to measure the
probe states.

"""
struct LSProcessTomo
    statecount::Int
    obscount::Int
    inputdim::Int
    outputdim::Int
    pred::Matrix{ComplexF64}
    function LSProcessTomo(states::Vector,obs::Vector)
        @assert all([LinearAlgebra.ishermitian(o) for o in obs]) "Observables must be Hermitian"
        @assert all([isstate(o) for o in states]) "States must be valid density matrices"

        pred = build_choi_process_predictor(states, obs)
        outputdim = size(pred,1)
        inputdim = size(pred,2)
        new(length(states),length(obs),inputdim,outputdim,pred)
    end
end

function predict(method::LSProcessTomo, process::Matrix)
    reshape(real(method.pred*vec(QuantumInfo.choi_liou_involution(process))), (method.statecount,method.obscount))
end

begin
    global fit
    tomo_dim = 0
    ptrb = SparseArrays.spzeros(0,0)

    function trb_sop(da,db)
        sop = zeros(da^2,(da*db)^2)
        for i=0:da-1
            for j=0:da-1
                for k=0:db-1
                    sop += vec(QuantumInfo.ketbra(i,j,da))*vec(kron(QuantumInfo.ketbra(i,j,da),QuantumInfo.ketbra(k,k,db)))'
                end
            end
        end
        return SparseArrays.sparse(sop)
    end

    function fit(method::LSProcessTomo,
                 means::Vector{Float64},
                 vars = ones(length(means));
                 #solver = MosekSolver(LOG=0))
                 solver = SCS.SCSSolver(verbose=0, max_iters=10_000, eps = 1e-8))

        if length(means) != length(vars) || size(method.pred,1) != length(means)
            error("Size of observations and predictons do not match.")
        end
        d4 = size(method.pred,2)
        d2 = Int(sqrt(d4))
        d  = Int(sqrt(d2))

        # We assume that the predictions are always real-valued
        # and we need to do the complex->real translation manually since
        # Convex.jl does not support complex numbers yet
        rpred = real(method.pred)
        ipred = -imag(method.pred)
        ivars = 1 ./ sqrt.(vars)

        if d != tomo_dim
            ptrb = trb_sop(d,d)
        end

        ρr = Convex.Variable(d2,d2)
        ρi = Convex.Variable(d2,d2)

        # least squares problem
#       NOTE: Notation "^2" is not working here.
#        problem = Convex.minimize( Convex.norm( (means - (rpred*vec(ρr) + ipred*vec(ρi))) .* ivars, 2 )^2 )
        problem = Convex.minimize( Convex.square(Convex.norm( (means - (rpred*vec(ρr) + ipred*vec(ρi))) .* ivars, 2 ) ))

        # CP constraint
        problem.constraints += Convex.isposdef([ρr ρi; -ρi ρr])
        # TP constraint
        problem.constraints += reshape(ptrb*vec(ρr),d,d) == QuantumInfo.eye(d)
        problem.constraints += reshape(ptrb*vec(ρi),d,d) == zeros(d,d)

        Convex.solve!(problem, solver)

        return QuantumInfo.choi_liou_involution((ρr.value + 1im*ρi.value)), problem.optval, problem.status
    end
end

function fit(method::LSProcessTomo,
             means::Matrix,
             vars::Matrix{Float64}=ones(size(means)))
    if size(means) != (method.statecount, method.obscount)
        error("Matrix of sample means must be $((method.statecount, method.obscount)), but is $(size(means))")
    end
    if size(vars) != (method.statecount, method.obscount)
        error("Matrix of sample variances must be $((method.statecount, method.obscount)), but is $(size(means))")
    end
    fit(method, vec(means), vec(vars))
end
