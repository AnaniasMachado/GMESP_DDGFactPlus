
using LinearAlgebra

# Compute F such that C - psi*I = F*F'.
function compute_F(
    C::Symmetric{<:Real,<:AbstractMatrix};
    psi::Float64 = 0.0,
    atol::Float64 = 1e-8,
)
    psi = max(0.0, psi)

    Cpsi = Symmetric(Matrix(C) - psi * I)
    lambda, Q = eigen(Cpsi)

    perm = sortperm(lambda, rev = true)
    lambda = lambda[perm]
    Q = Q[:, perm]

    lambda = map(v -> abs(v) <= atol ? 0.0 : v, lambda)

    F = Q * Diagonal(sqrt.(lambda))

    return F
end

# Compute M_psi(x) = F' * Diagonal(x) * F.
function M_psi(
    x::Vector{Float64},
    F::AbstractMatrix{Float64},
)
    length(x) == size(F, 1) ||
        error("Dimension mismatch: length(x) must equal the number of rows of F.")

    return Symmetric(F' * (F .* reshape(x, :, 1)))
end

# Find iota and the corresponding averaged eigenvalue.
function find_iota(
    y::Vector{Float64},
    t::Int;
    atol::Float64 = 1e-10,
)
    1 <= t <= length(y) ||
        error("t must satisfy 1 <= t <= length(y).")

    tail_sum = sum(y)
    mid = tail_sum / t

    if mid >= y[1] - atol
        return 0, mid
    end

    for iota in 1:(t - 1)
        tail_sum -= y[iota]
        mid = tail_sum / (t - iota)

        if y[iota] > mid + atol && mid >= y[iota + 1] - atol
            return iota, mid
        end
    end

    error("No valid iota found. Check that y is sorted and nonnegative.")
end

# Evaluate f_t(M_psi(x); psi) from F.
function f_t_from_F(
    x::Vector{Float64},
    F::AbstractMatrix{Float64},
    t::Int,
    psi::Float64,
)
    M = M_psi(x, F)
    lambda = sort(eigvals(M), rev = true)

    return sum(log.(lambda[1:t] .+ psi))
end

# Evaluate f_t(M_psi(x); psi) from C.
function f_t(
    x::Vector{Float64},
    C::Symmetric{<:Real,<:AbstractMatrix},
    t::Int,
    psi::Float64,
)
    F = compute_F(C; psi = psi)

    return f_t_from_F(x, F, t, psi)
end

# Evaluate Gamma_t(M_psi(x); psi) from F.
function Gamma_t_from_F(
    x::Vector{Float64},
    F::AbstractMatrix{Float64},
    t::Int,
    psi::Float64,
)
    M = M_psi(x, F)
    lambda = sort(eigvals(M), rev = true)

    y = copy(lambda)
    y[1:t] .+= psi

    iota, mid = find_iota(y, t)

    val = iota > 0 ? sum(log.(y[1:iota])) : 0.0
    val += (t - iota) * log(mid)

    return val
end

# Evaluate Gamma_t(M_psi(x); psi) from C.
function Gamma_t(
    x::Vector{Float64},
    C::Symmetric{<:Real,<:AbstractMatrix},
    t::Int,
    psi::Float64,
)
    F = compute_F(C; psi = psi)

    return Gamma_t_from_F(x, F, t, psi)
end

# Evaluate the DDGFact objective at a fixed x.
function DDGFact_value_at_x(
    x::Vector{Float64},
    C::Symmetric{<:Real,<:AbstractMatrix},
    t::Int;
    atol::Float64 = 1e-8,
)
    F = compute_F(C; psi = 0.0, atol = atol)

    return Gamma_t_from_F(x, F, t, 0.0)
end

# Evaluate the DDGFact+ objective at a fixed x.
function DDGFactplus_value_at_x(
    x::Vector{Float64},
    C::Symmetric{<:Real,<:AbstractMatrix},
    t::Int,
    psi::Float64;
    atol::Float64 = 1e-8,
)
    F = compute_F(C; psi = psi, atol = atol)

    return Gamma_t_from_F(x, F, t, psi)
end

# Compute the spectral upper bound.
function spectral_bound(
    C::Symmetric{<:Real,<:AbstractMatrix},
    t::Int,
)
    lambda = reverse(eigvals(C))

    return sum(log, @view lambda[1:t])
end

# Check the sufficient condition for a general-t closed-form solution.
function closed_form_general(
    C::Symmetric{<:Real,<:AbstractMatrix},
    s::Int,
    t::Int,
    psi::Float64;
    J1::AbstractVector{<:Integer} = Int[],
    J0::AbstractVector{<:Integer} = Int[],
    atol::Float64 = 1e-10,
)
    n = size(C, 1)

    1 <= t <= s <= n ||
        error("Need 1 <= t <= s <= n.")

    J1 = sort(unique(collect(J1)))
    J0 = sort(unique(collect(J0)))

    isempty(intersect(J1, J0)) ||
        error("J1 and J0 must be disjoint.")

    length(J1) <= s ||
        error("Too many variables fixed at one.")

    Jfree = setdiff(1:n, union(J1, J0))
    m = s - length(J1)

    m <= length(Jfree) ||
        error("Not enough free variables to satisfy cardinality s.")

    diagonal_C = diag(C)

    selected_free =
        m == 0 ? Int[] :
        Jfree[partialsortperm(diagonal_C[Jfree], 1:m; rev = true)]

    S_star = sort(vcat(J1, selected_free))

    x = zeros(Float64, n)
    x[S_star] .= 1.0

    Cpsi_S = Symmetric(Matrix(C)[S_star, S_star] - psi * I)

    lambda_max = eigmax(Cpsi_S)
    lambda_sum = sum(diagonal_C[S_star]) - s * psi

    valid = lambda_sum >= t * lambda_max - atol

    return valid, x, S_star
end

# Compute the trace upper bound with fixed variables.
function trace_upper_bound(
    C::Symmetric{<:Real,<:AbstractMatrix},
    s::Int,
    t::Int,
    psi::Float64;
    F1::Vector{Int} = Int[],
    F0::Vector{Int} = Int[],
)
    n = size(C, 1)

    Jfree = setdiff(1:n, union(F1, F0))
    m = s - length(F1)

    0 <= m <= length(Jfree) || return -Inf

    diagonal_C = diag(C)

    selected_free =
        m == 0 ? Int[] :
        Jfree[partialsortperm(diagonal_C[Jfree], 1:m; rev = true)]

    S_star = vcat(F1, selected_free)

    trM = sum(diagonal_C[S_star]) - s * psi

    return t * log(trM / t + psi)
end

# Compute the trace upper bound using a precomputed diagonal ordering.
function trace_upper_bound_sorted(
    C::Symmetric{<:Real,<:AbstractMatrix},
    s::Int,
    t::Int,
    psi::Float64,
    diagonal_order::Vector{Int};
    F1::Vector{Int} = Int[],
    F0::Vector{Int} = Int[],
)
    n = size(C, 1)
    m = s - length(F1)

    0 <= m <= n - length(F1) - length(F0) || return -Inf

    trM = -s * psi

    for i in F1
        trM += C[i, i]
    end

    selected = 0

    if m > 0
        for i in diagonal_order
            i in F1 && continue
            i in F0 && continue

            trM += C[i, i]
            selected += 1

            selected == m && break
        end
    end

    return t * log(trM / t + psi)
end

# Compute a spectral supergradient of Gamma_t.
function spectral_subgradient_Gamma_t(
    M::AbstractMatrix{Float64},
    t::Int,
    psi::Float64;
    atol::Float64 = 1e-10,
)
    eig = eigen(Symmetric(M))
    perm = sortperm(eig.values, rev = true)

    lambda = eig.values[perm]
    Q = eig.vectors[:, perm]

    y = copy(lambda)
    y[1:t] .+= psi

    iota, mid = find_iota(y, t; atol = atol)

    k = length(lambda)
    g = zeros(Float64, k)

    if iota > 0
        g[1:iota] .= 1.0 ./ y[1:iota]
    end

    if iota < k
        g[(iota + 1):k] .= 1.0 / mid
    end

    G = Q * Diagonal(g) * Q'

    return Symmetric(G)
end

# Compute a supergradient of x -> Gamma_t(M_psi(x); psi) from F.
function x_subgradient_Gamma_t_from_F(
    x::Vector{Float64},
    F::AbstractMatrix{Float64},
    t::Int,
    psi::Float64;
    atol::Float64 = 1e-10,
)
    M = M_psi(x, F)
    G = spectral_subgradient_Gamma_t(M, t, psi; atol = atol)

    grad = vec(sum((F * G) .* F; dims = 2))

    return grad
end

# Compute a supergradient of x -> Gamma_t(M_psi(x); psi) from C.
function x_subgradient_Gamma_t(
    x::Vector{Float64},
    C::Symmetric{<:Real,<:AbstractMatrix},
    t::Int,
    psi::Float64;
    atol::Float64 = 1e-10,
)
    F = compute_F(C; psi = psi)

    return x_subgradient_Gamma_t_from_F(x, F, t, psi; atol = atol)
end

# Compute the largest psi satisfying the general-t closed-form certificate.
function largest_closed_form_certificate_psi(
    C::Symmetric{<:Real,<:AbstractMatrix},
    s::Int,
    t::Int;
    J1::AbstractVector{<:Integer} = Int[],
    J0::AbstractVector{<:Integer} = Int[],
    atol::Float64 = 1e-10,
)
    n = size(C, 1)

    1 <= t <= s <= n ||
        error("Need 1 <= t <= s <= n.")

    J1 = unique(collect(J1))
    J0 = unique(collect(J0))

    isempty(intersect(J1, J0)) ||
        error("J1 and J0 must be disjoint.")

    length(J1) <= s ||
        error("Too many variables fixed at one.")

    Jfree = setdiff(1:n, union(J1, J0))
    m = s - length(J1)

    m <= length(Jfree) ||
        error("Not enough free variables to satisfy cardinality s.")

    diagonal_C = diag(C)

    selected_free =
        m == 0 ? Int[] :
        Jfree[partialsortperm(diagonal_C[Jfree], 1:m; rev = true)]

    S_star = sort(vcat(J1, selected_free))

    x = zeros(Float64, n)
    x[S_star] .= 1.0

    lambda = sort(
        eigvals(Symmetric(Matrix(C)[S_star, S_star]));
        rev = true,
    )

    certificate_margin =
        sum(@view lambda[2:end]) - (t - 1) * lambda[1]

    if certificate_margin < -atol
        return false, NaN, x, S_star
    end

    lambda_min_C = max(0.0, eigmin(C))

    psi =
        s == t ?
        lambda_min_C :
        min(
            lambda_min_C,
            max(0.0, certificate_margin / (s - t)),
        )

    return true, psi, x, S_star
end
