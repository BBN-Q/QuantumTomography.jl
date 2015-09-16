#using QuantumTomography
include("../src/QuantumTomography.jl")

function test()
    obs = Matrix[[0 1; 1 0], [0 -1im; 1im 0], pauli(3), -[0 1; 1 0], -[0 -1im; 1im 0], -pauli(3)]
    
    st = StateTomography(obs)

    ρ = .98*projector(rand_pure_state(2))+.02*rand_mixed_state(2)

    ideal_means = predict(st,ρ)
        
    samples = [ rand(Bernoulli((μ+1)/2),n) for μ in ideal_means ]
    sample_mean = 2*map(mean,samples)-1
    sample_var  = 4*map(var,samples)/n
    
    ρest, obj, status = qst_ml(pred, sample_mean, sample_var);
    
    return status, snorm(ρ-ρest,1), obj, ρ, ρest
end

function test_qst_ml(n=1000)
    obs = Matrix[[0 1; 1 0], [0 -1im; 1im 0], pauli(3), -[0 1; 1 0], -[0 -1im; 1im 0], -pauli(3)]
    
    pred = reduce(vcat,[vec(o)' for o in obs])
    
    ρ = .98*projector(rand_pure_state(2))+.02*rand_mixed_state(2)
    
    ideal_means = real(pred*vec(ρ))
    
    samples = [ rand(Bernoulli((μ+1)/2),n) for μ in ideal_means ]
    sample_mean = 2*map(mean,samples)-1
    sample_var  = 4*map(var,samples)/n
    
    ρest, obj, status = qst_ml(pred, sample_mean, sample_var);
    
    return status, snorm(ρ-ρest,1), obj, ρ, ρest
end

function test_qst_ml_ideal(n=1000)
    obs = Matrix[[0 1; 1 0], [0 -1im; 1im 0], pauli(3), -[0 1; 1 0], -[0 -1im; 1im 0], -pauli(3)]
    
    pred = reduce(vcat,[vec(o)' for o in obs])
    
    ρ = .98*projector(rand_pure_state(2))+.02*rand_mixed_state(2)
    
    ideal_means = real(pred*vec(ρ))
    
    ps = (ideal_means+1)/2
    ideal_vars = 4*ps.*(1-ps)/n 
    
    ρest1, obj1, status1 = qst_ml(pred, ideal_means, ideal_vars);
    ρest2, obj2, status2 = qst_ml(obs, ideal_means, ideal_vars);
    
    return status1, status2, snorm(ρ-ρest1,1), snorm(ρ-ρest2,1), obj1-obj2, ρ, ρest1, ρest2
end

function test_qpt_ml(n=1000;ρ=zeros(Float64,0,0))

    obs  = Matrix[ [0 1; 1 0], [0 -1im; 1im 0], pauli(3) ]
    prep = map(projector,Vector[ [1;0], [0;1], 1/sqrt(2)*[1;1], 1/sqrt(2)*[1;-1], 1/sqrt(2)*[1;-1im], 1/sqrt(2)*[1;1im] ] )
    
    exps = Matrix[ choi_liou_involution(vec(o)*vec(p)') for o in obs, p in prep ]

    pred = reduce(vcat, map(m->vec(m)', vec(exps)) )
    
    if trace(ρ) == 0.0
        ρ = choi_liou_involution(liou(rand_unitary(2))) 
    end
    
    ideal_means = real(pred*vec(ρ))

    samples = [ rand(Bernoulli((μ+1)/2),n) for μ in ideal_means ]
    sample_mean = 2*map(mean,samples)-1
    sample_var  = 4*map(var,samples)/n
    
    ρest, obj, status = qpt_ml(pred, sample_mean, sample_var);
    
    # get the normalization right
    choi_err = ρ-ρest |> choi_liou_involution |> liou2choi

    println("Status                  : $(status)")
    println("Diamond norm lower bound: $(snorm(choi_err,1))")
    println("χ² error                : $(obj)")
    println("Eigvals ρ:")
    for ev in eigvals(ρ)
        println(ev)
    end
    println("Eigvals ρest:")
    for ev in eigvals(ρest)
        println(real(ev))
    end
    
    return status, snorm(choi_err,1), obj, abs(choi_err), eigvals(ρ), eigvals(ρest)
end

function test_trb(da,db)
    r = rand_mixed_state(da*db)
    norm(trace(r,[da,db],2)-mat(trb_sop(da,db)*vec(r)),1)
end
