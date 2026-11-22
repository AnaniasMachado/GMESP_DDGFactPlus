using LinearAlgebra
using Printf
using DataStructures: BinaryMaxHeap

const TraceBound = :Trace
const DDGFact = :DDGFact
const DDGFactplus = :DDGFactplus

struct GMESPNode
    F1::Vector{Int}
    F0::Vector{Int}
    ub::Float64
    x::Vector{Float64}
    keep::Vector{Int}
    depth::Int
end

Base.isless(a::GMESPNode, b::GMESPNode) = a.ub < b.ub

const BNB_TIMING = Dict{Symbol,Float64}()

# Normalize the relaxation name.
function _normalize_relaxation(relaxation)
    r = Symbol(relaxation)
    r in (:Trace, :DDGFact, :DDGFactplus) ||
        error("Unknown relaxation: $r")
    return r
end

# Validate the variable-fixing rule.
function _normalize_fixing_rule(rule::Symbol)
    rule in (:none, :dual, :primal, :both) ||
        error("Invalid fixing_rule: $rule")
    return rule
end

# Validate the B&B options.
function _validate_bnb_options(relaxation::Symbol, fixing_rule::Symbol)
    _normalize_relaxation(relaxation)
    _normalize_fixing_rule(fixing_rule)

    if relaxation == :Trace && fixing_rule != :none
        error("TraceBound supports only :none fixing.")
    end

    if relaxation == :DDGFact && fixing_rule in (:primal, :both)
        error("DDGFact supports only :none and :dual fixing.")
    end

    return nothing
end

# Evaluate the objective at a selected subset.
function _gmesp_obj(C::AbstractMatrix, S::Vector{Int}, t::Int)
    lambda = reverse(eigvals(Symmetric(Matrix(C[S, S]))))
    return sum(log, max.(lambda[1:t], 1e-30))
end

# Recover a subset from a relaxation solution.
function _subset_value(
    C::Symmetric,
    s::Int,
    t::Int,
    keep::Vector{Int},
    x::Vector{Float64},
)
    (isempty(x) || length(keep) < s) && return -Inf, Int[]

    order = sortperm(x; rev = true)
    S = sort(keep[order[1:s]])

    value = try
        _gmesp_obj(C, S, t)
    catch
        -Inf
    end

    return value, S
end

# Check whether a relaxation solution is integral.
function _is_integer_point(x::AbstractVector)
    isempty(x) && return false
    return maximum(abs.(x .- round.(x))) <= 1e-6
end

# Choose a branching variable.
function _branch_var(
    C::Symmetric,
    keep::Vector{Int},
    x::Vector{Float64},
    F1::Vector{Int},
    relaxation::Symbol,
)
    fixed = Set(F1)

    if relaxation == :Trace ||
       (relaxation == :DDGFactplus && _is_integer_point(x))

        best_i = 0
        best_d = -Inf

        for i in keep
            i in fixed && continue

            if C[i, i] > best_d
                best_i = i
                best_d = C[i, i]
            end
        end

        return best_i
    end

    best_i = 0
    best_f = -Inf

    for k in eachindex(keep)
        i = keep[k]
        i in fixed && continue

        f = min(x[k], 1.0 - x[k])

        if f > best_f
            best_i = i
            best_f = f
        end
    end

    return best_i
end

# Map fixed-one indices to the reduced matrix.
function _local_fix1_indices(
    keep::Vector{Int},
    F1::Vector{Int},
)
    positions = Dict(i => k for (k, i) in enumerate(keep))
    return sort([positions[i] for i in F1])
end

# Choose a default value of psi.
function _default_psi(
    C::Symmetric{<:Real,<:AbstractMatrix};
    psi_margin::Float64 = 1e-7,
    psi_floor::Float64 = 0.0,
)
    lambda_min = eigmin(C)
    psi = max(psi_floor, lambda_min - psi_margin)

    0.0 < psi <= lambda_min ||
        error("DDGFact+ requires 0 < psi <= lambda_min(C).")

    return psi
end

# Reset the B&B timing counters.
function _reset_bnb_timing!()
    empty!(BNB_TIMING)

    for key in (
        :knitro_time,
        :relaxation_solve_time,
        :factorization_time,
        :dual_solution_time,
        :variable_fixing_time,
        :variable_fixing_calls,
        :open_list_time,
        :node_setup_time,
        :bound_computation_time,
        :closed_form_time,
        :closed_form_uses,
        :trace_prunes,
    )
        BNB_TIMING[key] = 0.0
    end

    return nothing
end

# Accumulate a timing measurement.
function _add_bnb_timing!(key::Symbol, value::Real)
    BNB_TIMING[key] = get(BNB_TIMING, key, 0.0) + Float64(value)
    return nothing
end

# Return a copy of the timing counters.
function _copy_bnb_timing()
    return copy(BNB_TIMING)
end

# Construct a standard node result.
function _node_result(
    relaxation::Symbol,
    keep::Vector{Int},
    x::Vector{Float64},
    ub::Float64;
    determined::Bool = false,
    dual_solution = nothing,
    l::Vector{Float64} = Float64[],
    c::Vector{Float64} = Float64[],
    psi = nothing,
    F = nothing,
    primal_obj::Float64 = NaN,
    closed_form::Bool = false,
)
    return (
        ub = ub,
        x = x,
        keep = keep,
        determined = determined,
        relaxation = relaxation,
        dual_solution = dual_solution,
        l = l,
        c = c,
        psi = psi,
        F = F,
        primal_obj = primal_obj,
        closed_form = closed_form,
    )
end

# Construct an infeasible node result.
function _infeasible_node_return(
    keep::Vector{Int},
    relaxation::Symbol,
)
    return _node_result(
        relaxation,
        keep,
        Float64[],
        -Inf,
    )
end

# Construct the exact result for a determined node.
function _determined_node(
    C::Symmetric,
    s::Int,
    t::Int,
    keep::Vector{Int},
    F1::Vector{Int},
    relaxation::Symbol;
    psi::Union{Nothing,Float64} = nothing,
)
    S = length(F1) == s ? sort(F1) : copy(keep)
    selected = Set(S)
    x = [i in selected ? 1.0 : 0.0 for i in keep]

    ub = _gmesp_obj(C, S, t)

    return _node_result(
        relaxation,
        keep,
        x,
        ub;
        determined = true,
        primal_obj = ub,
        psi = relaxation == :DDGFact ? 0.0 : psi,
    )
end

# Bound a node using the selected relaxation.
function _gmesp_node(
    C::Symmetric,
    s::Int,
    t::Int,
    F1::Vector{Int},
    F0::Vector{Int};
    relaxation = DDGFact,
    psi::Union{Nothing,Float64} = nothing,
    cf_trace::Bool = false,
    psi_trace::Union{Nothing,Float64} = nothing,
    diagonal_order::Union{Nothing,Vector{Int}} = nothing,
    atol::Float64 = 1e-8,
    psi_margin::Float64 = 1e-7,
    psi_floor::Float64 = 0.0,
)
    relaxation = _normalize_relaxation(relaxation)

    keep = Int[]
    infeasible = false

    setup_time = @elapsed begin
        keep = setdiff(1:size(C, 1), F0)
        infeasible =
            !isempty(intersect(F1, F0)) ||
            length(F1) > s ||
            length(keep) < s
    end

    _add_bnb_timing!(:node_setup_time, setup_time)

    infeasible &&
        return _infeasible_node_return(keep, relaxation)

    if length(F1) == s || length(keep) == s
        return _determined_node(
            C, s, t, keep, F1, relaxation;
            psi = psi,
        )
    end

    if relaxation == :Trace
        return _bound_trace_node(
            C, s, t, F1, F0, keep;
            psi = psi,
            diagonal_order = diagonal_order,
            atol = atol,
        )
    end

    Ck = Symmetric(Matrix(C[keep, keep]))
    fix1 = _local_fix1_indices(keep, F1)

    l = zeros(Float64, length(keep))
    c = ones(Float64, length(keep))
    l[fix1] .= 1.0

    return _bound_relaxation_node(
        Val(relaxation),
        Ck,
        s,
        t,
        fix1,
        keep,
        l,
        c;
        psi = psi,
        cf_trace = cf_trace,
        atol = atol,
        psi_margin = psi_margin,
        psi_floor = psi_floor,
    )
end

# Apply the relaxation-specific variable-fixing rules.
function variable_fixing_relaxation_soln(
    F1::Vector{Int},
    F0::Vector{Int},
    r,
    s::Int,
    t::Int,
    lb::Float64;
    fixing_rule::Symbol = :dual,
    tol::Real = 1e-9,
)
    fixing_rule = _normalize_fixing_rule(fixing_rule)

    if fixing_rule == :none ||
       r.relaxation == :Trace ||
       r.dual_solution === nothing
        return sort(F1), sort(F0)
    end

    _validate_bnb_options(r.relaxation, fixing_rule)

    fixing = _fixing_relaxation(
        Val(r.relaxation),
        r,
        s,
        t,
        lb;
        fixing_rule = fixing_rule,
        tol = Float64(tol),
    )

    F1new = sort(union(F1, r.keep[fixing.fix_one]))
    F0new = sort(union(F0, r.keep[fixing.fix_zero]))

    return F1new, F0new
end

# Initialize the incumbent using local search.
function _initial_incumbent(
    C::Symmetric,
    s::Int,
    t::Int,
)
    x_ls, lb = run_all_LS(C, s, t)
    S_inc = sort(findall(x_ls .> 0.5))
    return x_ls, lb, S_inc
end

# Register an integer relaxation solution.
function _register_integer_solution!(
    C::Symmetric,
    s::Int,
    t::Int,
    r,
    state::Base.RefValue,
)
    _is_integer_point(r.x) || return nothing

    true_obj, S = _subset_value(C, s, t, r.keep, r.x)
    isfinite(true_obj) || return nothing

    st = state[]
    int_gap = r.ub - true_obj

    lb = st.lb
    S_inc = st.S_inc
    int_gap_inc = st.int_gap_inc

    if true_obj > lb
        lb = true_obj
        S_inc = S
        int_gap_inc = int_gap
    end

    state[] = (
        lb = lb,
        S_inc = S_inc,
        n_int_sols = st.n_int_sols + 1,
        int_gap_sum = st.int_gap_sum + int_gap,
        int_gap_max = max(st.int_gap_max, int_gap),
        int_gap_inc = int_gap_inc,
    )

    return nothing
end

# Add a node to the open list if its bound survives.
function _push_if_survives!(
    open,
    F1::Vector{Int},
    F0::Vector{Int},
    r,
    lb::Float64,
    depth::Int,
)
    r.determined && return false
    r.ub <= lb && return false

    push!(
        open,
        GMESPNode(F1, F0, r.ub, r.x, r.keep, depth),
    )

    return true
end

# Bound and process a child node.
function _bound_child!(
    C::Symmetric,
    s::Int,
    t::Int,
    F1::Vector{Int},
    F0::Vector{Int},
    parent::GMESPNode,
    open,
    state::Base.RefValue,
    counters::Base.RefValue,
    node_kwargs;
    fixing_rule::Symbol,
    atol::Float64,
)
    depth = parent.depth + 1

    if node_kwargs.relaxation == :Trace
        r = _gmesp_node(C, s, t, F1, F0; node_kwargs...)

        cnt = counters[]
        counters[] = (
            nodes = cnt.nodes + 1,
            nfix0 = cnt.nfix0,
            nfix1 = cnt.nfix1,
        )

        _register_integer_solution!(C, s, t, r, state)
        _push_if_survives!(open, F1, F0, r, state[].lb, depth)

        return nothing
    end

    if node_kwargs.cf_trace
        ub_trace = trace_upper_bound(
            C,
            s,
            t,
            node_kwargs.psi_trace;
            F1 = F1,
            F0 = F0,
        )

        if ub_trace <= state[].lb
            _add_bnb_timing!(:trace_prunes, 1)
            return nothing
        end
    end

    r = _gmesp_node(C, s, t, F1, F0; node_kwargs...)

    cnt = counters[]
    counters[] = (
        nodes = cnt.nodes + 1,
        nfix0 = cnt.nfix0,
        nfix1 = cnt.nfix1,
    )

    _register_integer_solution!(C, s, t, r, state)

    (r.determined || r.ub <= state[].lb) && return nothing

    F1new = F1
    F0new = F0

    fixing_time = @elapsed begin
        F1new, F0new = variable_fixing_relaxation_soln(
            F1,
            F0,
            r,
            s,
            t,
            state[].lb;
            fixing_rule = fixing_rule,
            tol = atol,
        )
    end

    _add_bnb_timing!(:variable_fixing_time, fixing_time)
    _add_bnb_timing!(:variable_fixing_calls, 1)

    if length(F1new) > length(F1) || length(F0new) > length(F0)
        cnt = counters[]

        counters[] = (
            nodes = cnt.nodes,
            nfix0 = cnt.nfix0 + length(F0new) - length(F0),
            nfix1 = cnt.nfix1 + length(F1new) - length(F1),
        )

        if node_kwargs.cf_trace
            ub_trace = trace_upper_bound(
                C,
                s,
                t,
                node_kwargs.psi_trace;
                F1 = F1new,
                F0 = F0new,
            )

            if ub_trace <= state[].lb
                _add_bnb_timing!(:trace_prunes, 1)
                return nothing
            end
        end

        r = _gmesp_node(
            C, s, t, F1new, F0new;
            node_kwargs...,
        )

        _register_integer_solution!(C, s, t, r, state)

        (r.determined || r.ub <= state[].lb) && return nothing

        F1 = F1new
        F0 = F0new
    end

    _push_if_survives!(
        open,
        F1,
        F0,
        r,
        state[].lb,
        depth,
    )

    return nothing
end

# Print the root-node summary.
function _print_root_log(
    relaxation::Symbol,
    fixing_rule::Symbol,
    lb::Float64,
    ub::Float64,
)
    @printf(
        "root: relaxation = %s  fixing = %s  lb = %.4f  ub = %.4f  gap = %.4f\n",
        String(relaxation),
        String(fixing_rule),
        lb,
        ub,
        ub - lb,
    )
    flush(stdout)
    return nothing
end

# Print the current B&B progress.
function _print_progress(
    nodes::Int,
    open,
    nfix0::Int,
    nfix1::Int,
    lb::Float64,
)
    ub = isempty(open) ? lb : max(lb, first(open).ub)

    @printf(
        "nodes = %d  open = %d  fix0 = %d  fix1 = %d  lb = %.4f  ub = %.4f  gap = %.4f\n",
        nodes,
        length(open),
        nfix0,
        nfix1,
        lb,
        ub,
        ub - lb,
    )

    flush(stdout)
    return nothing
end

# Print the final B&B summary.
function _print_final_log(
    status::String,
    relaxation::Symbol,
    fixing_rule::Symbol,
    nodes::Int,
    nfix0::Int,
    nfix1::Int,
    lb::Float64,
    ub::Float64,
    n_int_sols::Int,
    int_gap_max::Float64,
    int_gap_avg::Float64,
    int_gap_opt::Float64,
)
    @printf(
        "done [%s]. relaxation = %s  fixing = %s  nodes = %d  fix0 = %d  fix1 = %d  lb = %.4f  ub = %.4f  gap = %.4f\n",
        status,
        String(relaxation),
        String(fixing_rule),
        nodes,
        nfix0,
        nfix1,
        lb,
        ub,
        ub - lb,
    )

    @printf(
        "integer solns = %d  max int-gap = %.4f  avg int-gap = %.4f  int-gap@opt = %.4f\n",
        n_int_sols,
        int_gap_max,
        int_gap_avg,
        int_gap_opt,
    )

    flush(stdout)
    return nothing
end

# Solve GMESP using branch-and-bound.
function solve_bnb_gmesp(
    C::Symmetric,
    s::Int,
    t::Int;
    relaxation = DDGFact,
    fixing_rule::Symbol = relaxation == TraceBound ? :none : :dual,
    psi::Union{Nothing,Float64} = nothing,
    cf_trace::Bool = false,
    time_limit::Real = 3600.0,
    verbose::Bool = true,
    atol::Float64 = 1e-8,
    psi_margin::Float64 = 1e-7,
    psi_floor::Float64 = 0.0,
)
    n = size(C, 1)
    @assert 1 <= t <= s < n

    relaxation = _normalize_relaxation(relaxation)
    fixing_rule = _normalize_fixing_rule(fixing_rule)
    _validate_bnb_options(relaxation, fixing_rule)

    diagonal_order = nothing

    if relaxation == :Trace
        psi = _trace_psi(C, psi; atol = atol)
        diagonal_order = sortperm(diag(C); rev = true)
    end

    _, lb0, S_inc0 = _initial_incumbent(C, s, t)

    _reset_bnb_timing!()

    state = Ref((
        lb = lb0,
        S_inc = S_inc0,
        n_int_sols = 0,
        int_gap_sum = 0.0,
        int_gap_max = 0.0,
        int_gap_inc = NaN,
    ))

    counters = Ref((
        nodes = 1,
        nfix0 = 0,
        nfix1 = 0,
    ))

    psi_trace = nothing

    if cf_trace && relaxation != :Trace
        psi_trace =
            psi === nothing ?
            max(0.0, eigmin(C) - psi_margin) :
            psi
    end

    node_kwargs = (
        relaxation = relaxation,
        psi = psi,
        cf_trace = cf_trace,
        psi_trace = psi_trace,
        diagonal_order = diagonal_order,
        atol = atol,
        psi_margin = psi_margin,
        psi_floor = psi_floor,
    )

    r_root = _gmesp_node(
        C,
        s,
        t,
        Int[],
        Int[];
        node_kwargs...,
    )

    ub_root = r_root.ub

    _register_integer_solution!(C, s, t, r_root, state)

    verbose && _print_root_log(
        relaxation,
        fixing_rule,
        state[].lb,
        ub_root,
    )

    open = BinaryMaxHeap{GMESPNode}()

    _push_if_survives!(
        open,
        Int[],
        Int[],
        r_root,
        state[].lb,
        0,
    )

    t0 = time()
    time_limit_hit = false
    next_report = 1000

    while !isempty(open)
        if time() - t0 >= time_limit
            time_limit_hit = true
            break
        end

        node = nothing

        open_list_time = @elapsed begin
            node = pop!(open)
        end

        _add_bnb_timing!(:open_list_time, open_list_time)

        node.ub <= state[].lb && continue

        i = _branch_var(
            C,
            node.keep,
            node.x,
            node.F1,
            relaxation,
        )

        i == 0 && continue

        F1child = sort(vcat(node.F1, i))

        _bound_child!(
            C,
            s,
            t,
            F1child,
            node.F0,
            node,
            open,
            state,
            counters,
            node_kwargs;
            fixing_rule = fixing_rule,
            atol = atol,
        )

        F0child = sort(vcat(node.F0, i))

        _bound_child!(
            C,
            s,
            t,
            node.F1,
            F0child,
            node,
            open,
            state,
            counters,
            node_kwargs;
            fixing_rule = fixing_rule,
            atol = atol,
        )

        cnt = counters[]

        if verbose && cnt.nodes >= next_report
            _print_progress(
                cnt.nodes,
                open,
                cnt.nfix0,
                cnt.nfix1,
                state[].lb,
            )

            next_report += 1000
        end
    end

    st = state[]
    cnt = counters[]

    tree_exhausted = isempty(open)

    final_ub = tree_exhausted ?
        st.lb :
        max(st.lb, first(open).ub)

    status = time_limit_hit ? "TIME LIMIT" : "OPTIMAL"

    int_gap_avg =
        st.n_int_sols == 0 ?
        NaN :
        st.int_gap_sum / st.n_int_sols

    verbose && _print_final_log(
        status,
        relaxation,
        fixing_rule,
        cnt.nodes,
        cnt.nfix0,
        cnt.nfix1,
        st.lb,
        final_ub,
        st.n_int_sols,
        st.int_gap_max,
        int_gap_avg,
        st.int_gap_inc,
    )

    timing = _copy_bnb_timing()

    stats = (
        nodes = cnt.nodes,
        lb = st.lb,
        ub = final_ub,
        gap = final_ub - st.lb,
        root_ub = ub_root,
        nfix0 = cnt.nfix0,
        nfix1 = cnt.nfix1,
        n_int_sols = st.n_int_sols,
        int_gap_max = st.int_gap_max,
        int_gap_avg = int_gap_avg,
        int_gap_opt = st.int_gap_inc,
        wall_time = time() - t0,
        time_limit_hit = time_limit_hit,
        tree_exhausted = tree_exhausted,
        relaxation = relaxation,
        fixing_rule = fixing_rule,
        cf_trace = cf_trace,
        closed_form_uses =
            Int(round(get(timing, :closed_form_uses, 0.0))),
        closed_form_time =
            get(timing, :closed_form_time, 0.0),
        trace_prunes =
            Int(round(get(timing, :trace_prunes, 0.0))),
        knitro_time =
            get(timing, :knitro_time, 0.0),
        relaxation_solve_time =
            get(timing, :relaxation_solve_time, 0.0),
        factorization_time =
            get(timing, :factorization_time, 0.0),
        bound_computation_time =
            get(timing, :bound_computation_time, 0.0),
        open_list_time =
            get(timing, :open_list_time, 0.0),
        node_setup_time =
            get(timing, :node_setup_time, 0.0),
        dual_solution_time =
            get(timing, :dual_solution_time, 0.0),
        variable_fixing_direct_time =
            get(timing, :variable_fixing_time, 0.0),
        variable_fixing_time =
            get(timing, :variable_fixing_time, 0.0),
        variable_fixing_calls =
            Int(round(get(timing, :variable_fixing_calls, 0.0))),
    )

    return st.S_inc, stats
end