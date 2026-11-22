using LinearAlgebra

# Choose a valid shift for the trace upper bound.
function _trace_psi(
    C::Symmetric,
    psi::Union{Nothing,Float64};
    atol::Float64 = 1e-8,
)
    lambda_min = eigmin(C)
    shift = psi === nothing ? max(0.0, lambda_min - atol) : psi

    0.0 <= shift <= lambda_min ||
        error("TraceBound requires 0 <= psi <= eigmin(C).")

    return shift
end

# Compute the trace upper bound with a preordered diagonal.
function _trace_upper_bound_sorted(
    C::Symmetric,
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

    for i in diagonal_order
        selected == m && break
        (i in F1 || i in F0) && continue

        trM += C[i, i]
        selected += 1
    end

    selected == m || return -Inf

    argument = trM / t + psi

    return argument > 0.0 ? t * log(argument) : -Inf
end

# Bound a TraceBound node using the largest available diagonal entries.
function _bound_trace_node(
    C::Symmetric,
    s::Int,
    t::Int,
    F1::Vector{Int},
    F0::Vector{Int},
    keep::Vector{Int};
    psi::Union{Nothing,Float64} = nothing,
    diagonal_order::Union{Nothing,Vector{Int}} = nothing,
    atol::Float64 = 1e-8,
)
    shift = psi === nothing ?
        _trace_psi(C, nothing; atol = atol) :
        psi

    order = diagonal_order === nothing ?
        sortperm(diag(C); rev = true) :
        diagonal_order

    ub = -Inf

    bound_time = @elapsed begin
        ub = _trace_upper_bound_sorted(
            C,
            s,
            t,
            shift,
            order;
            F1 = F1,
            F0 = F0,
        )
    end

    _add_bnb_timing!(:bound_computation_time, bound_time)

    return _node_result(
        :Trace,
        keep,
        Float64[],
        ub;
        psi = shift,
    )
end