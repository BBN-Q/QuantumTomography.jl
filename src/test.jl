using Convex, SCS, SchattenNorms

set_default_solver(SCSSolver(verbose=2))

K = 1000
n = 2

A = randn(n, n);
U,_,_ = svd(randn(n,n))
p = 0.0
ρ = (1-p)*U*[1 0; 0 0]*U'+p * QuantumInfo.eye(2)/2

nzm = round(Int, Convex.tr(ρ*[1 0; 0 0]) * K |> real)
nxm = round(Int, Convex.tr(ρ*[1 -1; -1 1]/2) * K |> real)

println("(x,z) = ($nxm,$nzm)")

x = Convex.Semidefinite(n)

constraints = Convex.tr(x) == 1

β = 0.05
problem = Convex.maximize( nzm * log(Convex.tr(x*[1 0; 0 0])) +
                    (K-nzm) * log(Convex.tr(x*[0 0; 0 1])) +
                    nxm * log(Convex.tr(x*[1 -1; -1 1])/2) +
                    (K-nxm) * log(Convex.tr(x*[1 1; 1 1])/2) +
                    # β * logdet(x),
                    β * log(lambda_min(x)),
                    constraints )

# Solve the problem by calling solve!
Convex.solve!(problem)

# Check the status of the problem
println(problem.status) # :Optimal, :Infeasible, :Unbounded etc.

# Get the optimum value
println(problem.optval)

#println(x.value)

println(round(Int, Convex.tr(x.value*[1 0; 0 0]) * K))
println(round(Int, Convex.tr(x.value*[1 -1; -1 1]/2) * K))

objective(x_) = nzm * log(Convex.tr(x_*[1 0; 0 0])) +
                (K-nzm) * log(Convex.tr(x_*[0 0; 0 1])) +
                nxm * log(Convex.tr(x_*[1 -1; -1 1])/2) +
                (K-nxm) * log(Convex.tr(x_*[1 1; 1 1])/2) +
                β * minimum(eigvals(x_))

println(ρ)
println(x.value)
println(trnorm(ρ-x.value)/2)
