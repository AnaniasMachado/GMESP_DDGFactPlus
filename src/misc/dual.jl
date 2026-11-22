
using LinearAlgebra

# Construct Theta from a feasible DDGFact solution.
function construct_Theta_from_x_DDGFact(
    xhat::Vector{Float64},
    F::AbstractMatrix{Float64},
    t::Int;
    epsilon::Float64 = 1e-6,
    atol::Float64 = 1e-8,
)
    M = M_psi(xhat, F)

    eig = eigen(Symmetric(M))
    perm = sortperm(eig.values, rev = true)

    lambda = eig.values[perm]
    U = eig.vectors[:, perm]

    k = length(lambda)

    1 <= t <= k ||
        error("t must satisfy 1 <= t <= size(M, 1).")

    rank_hat = count(v -> v > atol, lambda)

    rank_hat >= t ||
        error("M_psi(xhat) must have rank at least t.")

    iota, delta = find_iota(lambda, t; atol = atol)

    beta = zeros(Float64, k)

    if iota > 0
        beta[1:iota] .= 1.0 ./ lambda[1:iota]
    end

    if iota < rank_hat
        beta[(iota + 1):rank_hat] .= 1.0 / delta
    end

    if rank_hat < k
        beta[(rank_hat + 1):k] .= (1.0 + epsilon) / delta
    end

    Theta = Symmetric(U * Diagonal(beta) * U')

    return Theta, beta, lambda, iota, delta, rank_hat
end

# Solve the linear subproblem G(Theta) and construct its dual variables.
function solve_GTheta_no_side_constraints(
    d::Vector{Float64},
    s::Int;
    l::Vector{Float64} = zeros(length(d)),
    c::Vector{Float64} = ones(length(d)),
    atol::Float64 = 1e-8,
)
    n = length(d)

    if length(l) != n || length(c) != n
        error("l, c, and d must have the same length.")
    end

    if any(l .> c .+ atol)
        error("Bounds must satisfy l <= c.")
    end

    if sum(l) > s + atol || sum(c) < s - atol
        error("The fixed-variable polytope is empty.")
    end

    sigma = sortperm(d, rev = true)
    phi = 0

    for j in 1:n
        lhs = sum(c[sigma[1:j]]) + sum(l[sigma[(j + 1):n]])

        if lhs <= s + atol
            phi = j
        else
            break
        end
    end

    P = sigma[1:phi]
    central = phi < n ? sigma[phi + 1] : nothing
    Q = phi + 2 <= n ? sigma[(phi + 2):n] : Int[]

    xstar = zeros(Float64, n)

    if !isempty(P)
        xstar[P] .= c[P]
    end

    if !isempty(Q)
        xstar[Q] .= l[Q]
    end

    if central !== nothing
        xstar[central] = s - sum(c[P]) - sum(l[Q])
    end

    tau = central === nothing ? 0.0 : d[central]

    upsilon = zeros(Float64, n)
    nu = zeros(Float64, n)

    if !isempty(P)
        nu[P] .= d[P] .- tau
    end

    if !isempty(Q)
        upsilon[Q] .= tau .- d[Q]
    end

    upsilon[abs.(upsilon) .<= atol] .= 0.0
    nu[abs.(nu) .<= atol] .= 0.0

    minimum(upsilon) >= -atol ||
        error("Constructed upsilon is not nonnegative.")

    minimum(nu) >= -atol ||
        error("Constructed nu is not nonnegative.")

    residual = d + upsilon - nu .- tau

    maximum(abs.(residual)) <= 1e-6 ||
        error("Dual equality constraint is not satisfied.")

    abs(sum(xstar) - s) <= 1e-6 ||
        error("Constructed primal solution does not satisfy sum(x) = s.")

    linear_value = -dot(upsilon, l) + dot(nu, c) + tau * s

    return (
        upsilon = upsilon,
        nu = nu,
        tau = tau,
        xstar = xstar,
        sigma = sigma,
        P = P,
        Q = Q,
        central = central,
        linear_value = linear_value,
        residual = residual,
    )
end

# Evaluate the DGFact dual objective.
function DGFact_objective_value(
    Theta::AbstractMatrix{Float64},
    upsilon::Vector{Float64},
    nu::Vector{Float64},
    tau::Float64,
    s::Int,
    t::Int;
    l::Vector{Float64} = zeros(length(upsilon)),
    c::Vector{Float64} = ones(length(upsilon)),
)
    theta_eigs = sort(eigvals(Symmetric(Theta)), rev = true)
    k = length(theta_eigs)

    1 <= t <= k ||
        error("t must satisfy 1 <= t <= size(Theta, 1).")

    smallest_t = theta_eigs[(k - t + 1):k]
    linear_part = -dot(upsilon, l) + dot(nu, c) + tau * s

    if any(smallest_t .<= 0.0)
        return (
            objective_value = Inf,
            spectral_part = Inf,
            linear_part = linear_part,
        )
    end

    spectral_part = -sum(log, smallest_t) - t

    return (
        objective_value = spectral_part + linear_part,
        spectral_part = spectral_part,
        linear_part = linear_part,
    )
end

# Construct a DGFact dual solution from a feasible DDGFact solution.
function DGFact_dual_solution_from_DDGFact_x(
    xhat::Vector{Float64},
    F::AbstractMatrix{Float64},
    s::Int,
    t::Int;
    l::Vector{Float64} = zeros(length(xhat)),
    c::Vector{Float64} = ones(length(xhat)),
    epsilon::Float64 = 1e-6,
    atol::Float64 = 1e-8,
)
    Theta, beta, lambda, iota, delta, rank_hat =
        construct_Theta_from_x_DDGFact(
            xhat,
            F,
            t;
            epsilon = epsilon,
            atol = atol,
        )

    d = vec(sum((F * Theta) .* F; dims = 2))

    sol_G = solve_GTheta_no_side_constraints(
        d,
        s;
        l = l,
        c = c,
        atol = atol,
    )

    obj = DGFact_objective_value(
        Theta,
        sol_G.upsilon,
        sol_G.nu,
        sol_G.tau,
        s,
        t;
        l = l,
        c = c,
    )

    return (
        Theta = Theta,
        upsilon = sol_G.upsilon,
        nu = sol_G.nu,
        tau = sol_G.tau,
        objective_value = obj.objective_value,
        spectral_part = obj.spectral_part,
        linear_part = obj.linear_part,
        d = d,
        xstar_GTheta = sol_G.xstar,
        beta = beta,
        lambda = lambda,
        iota = iota,
        delta = delta,
        rank_hat = rank_hat,
        sigma = sol_G.sigma,
        P = sol_G.P,
        Q = sol_G.Q,
        central = sol_G.central,
        residual = sol_G.residual,
    )
end

# Evaluate the function h_psi appearing in the DGFact+ dual.
function h_psi_value(
    w::Float64,
    psi::Float64;
    atol::Float64 = 1e-10,
)
    psi > 0.0 ||
        error("h_psi requires psi > 0.")

    if w <= 0.0
        return Inf
    end

    if w < 1.0 / psi - atol
        return -log(w) + psi * w - 1.0
    end

    return log(psi)
end

# Construct Theta from a feasible DDGFact+ solution.
function construct_Theta_from_x_DDGFactplus(
    xhat::Vector{Float64},
    F::AbstractMatrix{Float64},
    t::Int,
    psi::Float64;
    atol::Float64 = 1e-8,
)
    psi > 0.0 ||
        error("DGFact+ construction requires psi > 0.")

    Mhat = M_psi(xhat, F)

    eig = eigen(Symmetric(Mhat))
    perm = sortperm(eig.values, rev = true)

    lambda = eig.values[perm]
    Q = eig.vectors[:, perm]

    k = length(lambda)

    1 <= t <= k ||
        error("t must satisfy 1 <= t <= size(Mhat, 1).")

    lambda[abs.(lambda) .<= atol] .= 0.0

    y = copy(lambda)
    y[1:t] .+= psi

    iota, mid = find_iota(y, t; atol = atol)

    beta = zeros(Float64, k)

    if iota > 0
        beta[1:iota] .= 1.0 ./ y[1:iota]
    end

    if iota < k
        beta[(iota + 1):k] .= 1.0 / mid
    end

    Theta = Symmetric(Q * Diagonal(beta) * Q')

    return (
        Theta = Theta,
        beta = beta,
        lambda = lambda,
        y = y,
        iota = iota,
        mid = mid,
    )
end

# Evaluate the DGFact+ dual objective.
function DGFactplus_objective_value(
    Theta::AbstractMatrix{Float64},
    upsilon::Vector{Float64},
    nu::Vector{Float64},
    tau::Float64,
    s::Int,
    t::Int,
    psi::Float64;
    l::Vector{Float64} = zeros(length(upsilon)),
    c::Vector{Float64} = ones(length(upsilon)),
    atol::Float64 = 1e-10,
)
    theta_eigs = sort(eigvals(Symmetric(Theta)), rev = true)
    k = length(theta_eigs)

    1 <= t <= k ||
        error("t must satisfy 1 <= t <= size(Theta, 1).")

    smallest_t = theta_eigs[(k - t + 1):k]

    spectral_part = sum(
        h_psi_value(w, psi; atol = atol)
        for w in smallest_t
    )

    linear_part = -dot(upsilon, l) + dot(nu, c) + tau * s

    return (
        objective_value = spectral_part + linear_part,
        spectral_part = spectral_part,
        linear_part = linear_part,
    )
end

# Construct a DGFact+ dual solution from a feasible DDGFact+ solution.
function DGFactplus_dual_solution_from_DDGFactplus_x(
    xhat::Vector{Float64},
    F::AbstractMatrix{Float64},
    s::Int,
    t::Int,
    psi::Float64;
    l::Vector{Float64} = zeros(length(xhat)),
    c::Vector{Float64} = ones(length(xhat)),
    atol::Float64 = 1e-8,
)
    psi > 0.0 ||
        error("DGFact+ construction requires psi > 0.")

    n = length(xhat)

    if length(l) != n || length(c) != n
        error("xhat, l, and c must have the same length.")
    end

    if any(l .> c)
        error("Bounds must satisfy l <= c.")
    end

    if any((l .!= 0.0) .& (l .!= 1.0)) ||
       any((c .!= 0.0) .& (c .!= 1.0))
        error("Bounds l and c must be binary vectors.")
    end

    theta_data = construct_Theta_from_x_DDGFactplus(
        xhat,
        F,
        t,
        psi;
        atol = atol,
    )

    Theta = theta_data.Theta
    d = vec(sum((F * Theta) .* F; dims = 2))

    sol_G = solve_GTheta_no_side_constraints(
        d,
        s;
        l = l,
        c = c,
        atol = atol,
    )

    obj = DGFactplus_objective_value(
        Theta,
        sol_G.upsilon,
        sol_G.nu,
        sol_G.tau,
        s,
        t,
        psi;
        l = l,
        c = c,
        atol = atol,
    )

    return (
        Theta = Theta,
        upsilon = sol_G.upsilon,
        nu = sol_G.nu,
        tau = sol_G.tau,
        objective_value = obj.objective_value,
        spectral_part = obj.spectral_part,
        linear_part = obj.linear_part,
        d = d,
        xstar_GTheta = sol_G.xstar,
        beta = theta_data.beta,
        lambda = theta_data.lambda,
        y = theta_data.y,
        iota = theta_data.iota,
        mid = theta_data.mid,
        sigma = sol_G.sigma,
        P = sol_G.P,
        Q = sol_G.Q,
        central = sol_G.central,
        residual = sol_G.residual,
    )
end
