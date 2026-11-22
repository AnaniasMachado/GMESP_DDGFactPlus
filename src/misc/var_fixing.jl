using LinearAlgebra

# Apply the DGFact dual variable-fixing rule.
function var_fixing_from_DGFact(
    upsilon::Vector{Float64},
    nu::Vector{Float64},
    zeta::Float64,
    LB::Float64;
    l::Vector{Float64} = zeros(length(upsilon)),
    c::Vector{Float64} = ones(length(upsilon)),
    atol::Float64 = 1e-8,
)
    n = length(upsilon)

    if length(nu) != n || length(l) != n || length(c) != n
        error("upsilon, nu, l, and c must have the same length.")
    end

    if any(l .> c)
        error("Bounds must satisfy l <= c.")
    end

    if any((l .!= 0.0) .& (l .!= 1.0)) ||
       any((c .!= 0.0) .& (c .!= 1.0))
        error("Bounds l and c must be binary vectors.")
    end

    gap = zeta - LB

    fix_zero = findall(j -> upsilon[j] > gap + atol, 1:n)
    fix_one = findall(j -> nu[j] > gap + atol, 1:n)

    conflict_zero = [j for j in fix_zero if l[j] == 1.0]
    conflict_one = [j for j in fix_one if c[j] == 0.0]
    both_fixed = intersect(fix_zero, fix_one)

    l_new = copy(l)
    c_new = copy(c)

    c_new[fix_zero] .= 0.0
    l_new[fix_one] .= 1.0

    infeasible_bounds = any(l_new .> c_new)

    return (
        fix_zero = fix_zero,
        fix_one = fix_one,
        l_new = l_new,
        c_new = c_new,
        gap = gap,
        both_fixed = both_fixed,
        conflict_zero = conflict_zero,
        conflict_one = conflict_one,
        infeasible_bounds = infeasible_bounds,
    )
end

# Construct a DGFact dual certificate and apply its fixing rule.
function var_fixing_DDGFact_dual(
    xhat::Vector{Float64},
    F::AbstractMatrix{Float64},
    s::Int,
    t::Int,
    LB::Float64;
    l::Vector{Float64} = zeros(length(xhat)),
    c::Vector{Float64} = ones(length(xhat)),
    epsilon::Float64 = 1e-6,
    atol::Float64 = 1e-8,
)
    dual_sol = DGFact_dual_solution_from_DDGFact_x(
        xhat,
        F,
        s,
        t;
        l = l,
        c = c,
        epsilon = epsilon,
        atol = atol,
    )

    fixing = var_fixing_from_DGFact(
        dual_sol.upsilon,
        dual_sol.nu,
        dual_sol.objective_value,
        LB;
        l = l,
        c = c,
        atol = atol,
    )

    return (
        dual_solution = dual_sol,
        fixing = fixing,
    )
end

# Apply the DGFact+ dual variable-fixing rule.
function var_fixing_from_DGFactplus(
    upsilon::Vector{Float64},
    nu::Vector{Float64},
    zeta::Float64,
    LB::Float64;
    l::Vector{Float64} = zeros(length(upsilon)),
    c::Vector{Float64} = ones(length(upsilon)),
    atol::Float64 = 1e-8,
)
    return var_fixing_from_DGFact(
        upsilon,
        nu,
        zeta,
        LB;
        l = l,
        c = c,
        atol = atol,
    )
end

# Construct a DGFact+ dual certificate and apply its fixing rule.
function var_fixing_DDGFactplus_dual(
    xhat::Vector{Float64},
    F::AbstractMatrix{Float64},
    s::Int,
    t::Int,
    psi::Float64,
    LB::Float64;
    l::Vector{Float64} = zeros(length(xhat)),
    c::Vector{Float64} = ones(length(xhat)),
    atol::Float64 = 1e-8,
)
    dual_sol = DGFactplus_dual_solution_from_DDGFactplus_x(
        xhat,
        F,
        s,
        t,
        psi;
        l = l,
        c = c,
        atol = atol,
    )

    fixing = var_fixing_from_DGFactplus(
        dual_sol.upsilon,
        dual_sol.nu,
        dual_sol.objective_value,
        LB;
        l = l,
        c = c,
        atol = atol,
    )

    return (
        dual_solution = dual_sol,
        fixing = fixing,
    )
end

# Maximize a linear function over the binary-bounded cardinality polytope.
function max_linear_cardinality_binary_bounds(
    g::Vector{Float64},
    s::Int;
    l::Vector{Float64} = zeros(length(g)),
    c::Vector{Float64} = ones(length(g)),
    atol::Float64 = 1e-8,
)
    n = length(g)

    if length(l) != n || length(c) != n
        error("g, l, and c must have the same length.")
    end

    if any(l .> c)
        return (
            value = -Inf,
            xstar = zeros(Float64, n),
            feasible = false,
            selected_free = Int[],
        )
    end

    if any((l .!= 0.0) .& (l .!= 1.0)) ||
       any((c .!= 0.0) .& (c .!= 1.0))
        error("Bounds l and c must be binary vectors.")
    end

    N1 = findall(i -> l[i] == 1.0 && c[i] == 1.0, 1:n)
    Nf = findall(i -> l[i] == 0.0 && c[i] == 1.0, 1:n)

    m = s - length(N1)

    if m < 0 || m > length(Nf)
        return (
            value = -Inf,
            xstar = zeros(Float64, n),
            feasible = false,
            selected_free = Int[],
        )
    end

    selected_free =
        m == 0 ? Int[] : Nf[partialsortperm(g[Nf], 1:m; rev = true)]

    xstar = copy(l)
    xstar[selected_free] .= 1.0

    return (
        value = dot(g, xstar),
        xstar = xstar,
        feasible = true,
        selected_free = selected_free,
    )
end

# Fix DDGFact+ variables using affine upper bounds from a supergradient.
function var_fixing_DDGFactplus_primal(
    xhat::Vector{Float64},
    F::AbstractMatrix{Float64},
    s::Int,
    t::Int,
    psi::Float64,
    LB::Float64;
    l::Vector{Float64} = zeros(length(xhat)),
    c::Vector{Float64} = ones(length(xhat)),
    atol::Float64 = 1e-8,
)
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

    if sum(l) > s + atol || sum(c) < s - atol
        error("The fixed-variable polytope is empty: need sum(l) <= s <= sum(c).")
    end

    if abs(sum(xhat) - s) > 1e-5
        @warn "xhat does not satisfy e'x = s within tolerance." sum_xhat = sum(xhat) s = s
    end

    g = x_subgradient_Gamma_t_from_F(
        xhat,
        F,
        t,
        psi;
        atol = atol,
    )

    gamma_value = Gamma_t_from_F(xhat, F, t, psi)

    lin_root = max_linear_cardinality_binary_bounds(
        g,
        s;
        l = l,
        c = c,
        atol = atol,
    )

    if !lin_root.feasible
        error("The root/node fixed-variable polytope is infeasible.")
    end

    affine_constant = gamma_value - dot(g, xhat)
    UB = affine_constant + lin_root.value
    gap = UB - LB

    free = findall(j -> l[j] == 0.0 && c[j] == 1.0, 1:n)

    UB_if_zero = fill(-Inf, n)
    UB_if_one = fill(-Inf, n)

    loss_if_zero = fill(Inf, n)
    loss_if_one = fill(Inf, n)

    status_if_zero = fill(false, n)
    status_if_one = fill(false, n)

    fix_zero = Int[]
    fix_one = Int[]

    for j in free
        l_zero = copy(l)
        c_zero = copy(c)
        l_zero[j] = 0.0
        c_zero[j] = 0.0

        lin_zero = max_linear_cardinality_binary_bounds(
            g,
            s;
            l = l_zero,
            c = c_zero,
            atol = atol,
        )

        if lin_zero.feasible
            UB_if_zero[j] = affine_constant + lin_zero.value
            loss_if_zero[j] = UB - UB_if_zero[j]
            status_if_zero[j] = true

            if UB_if_zero[j] < LB - atol
                push!(fix_one, j)
            end
        else
            push!(fix_one, j)
        end

        l_one = copy(l)
        c_one = copy(c)
        l_one[j] = 1.0
        c_one[j] = 1.0

        lin_one = max_linear_cardinality_binary_bounds(
            g,
            s;
            l = l_one,
            c = c_one,
            atol = atol,
        )

        if lin_one.feasible
            UB_if_one[j] = affine_constant + lin_one.value
            loss_if_one[j] = UB - UB_if_one[j]
            status_if_one[j] = true

            if UB_if_one[j] < LB - atol
                push!(fix_zero, j)
            end
        else
            push!(fix_zero, j)
        end
    end

    both_fixed = intersect(fix_zero, fix_one)

    l_new = copy(l)
    c_new = copy(c)

    c_new[fix_zero] .= 0.0
    l_new[fix_one] .= 1.0

    infeasible_bounds = any(l_new .> c_new)

    return (
        fix_zero = fix_zero,
        fix_one = fix_one,
        l_new = l_new,
        c_new = c_new,
        UB = UB,
        LB = LB,
        gap = gap,
        gamma_value = gamma_value,
        g = g,
        linear_root_value = lin_root.value,
        root_linear_solution = lin_root.xstar,
        UB_if_zero = UB_if_zero,
        UB_if_one = UB_if_one,
        loss_if_zero = loss_if_zero,
        loss_if_one = loss_if_one,
        status_if_zero = status_if_zero,
        status_if_one = status_if_one,
        both_fixed = both_fixed,
        infeasible_bounds = infeasible_bounds,
    )
end