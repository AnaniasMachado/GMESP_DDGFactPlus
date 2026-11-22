
using LinearAlgebra

# Solve the DDGFact relaxation and construct its dual solution.
function _bound_ddgfact_node(
    Ck::Symmetric,
    s::Int,
    t::Int,
    fix1::Vector{Int},
    l::Vector{Float64},
    c::Vector{Float64};
    atol::Float64,
)
    x = Float64[]
    primal_obj = NaN

    relaxation_time = @elapsed begin
        x, primal_obj = ddfact_gmesp(
            Ck,
            s,
            t;
            J1 = fix1,
            atol = atol,
        )
    end

    _add_bnb_timing!(:relaxation_solve_time, relaxation_time)
    _add_bnb_timing!(:knitro_time, relaxation_time)

    F = nothing

    factorization_time = @elapsed begin
        F = compute_F(Ck; atol = atol)
    end

    _add_bnb_timing!(:factorization_time, factorization_time)

    dual_sol = nothing

    dual_time = @elapsed begin
        dual_sol = DGFact_dual_solution_from_DDGFact_x(
            x,
            F,
            s,
            t;
            l = l,
            c = c,
            atol = atol,
        )
    end

    _add_bnb_timing!(:dual_solution_time, dual_time)

    return x, primal_obj, F, dual_sol
end

# Bound a node using DDGFact.
function _bound_relaxation_node(
    ::Val{:DDGFact},
    Ck::Symmetric,
    s::Int,
    t::Int,
    fix1::Vector{Int},
    keep::Vector{Int},
    l::Vector{Float64},
    c::Vector{Float64};
    psi::Union{Nothing,Float64} = nothing,
    cf_trace::Bool = false,
    atol::Float64 = 1e-8,
    psi_margin::Float64 = 1e-7,
    psi_floor::Float64 = 0.0,
)
    if cf_trace
        valid = false
        x_cf = Float64[]

        cf_time = @elapsed begin
            valid, x_cf, _ = closed_form_general(
                Ck,
                s,
                t,
                0.0;
                J1 = fix1,
                atol = atol,
            )
        end

        _add_bnb_timing!(:closed_form_time, cf_time)

        if valid
            primal_obj = DDGFact_value_at_x(
                x_cf,
                Ck,
                t;
                atol = atol,
            )

            F = nothing

            factorization_time = @elapsed begin
                F = compute_F(Ck; atol = atol)
            end

            _add_bnb_timing!(
                :factorization_time,
                factorization_time,
            )

            dual_sol = nothing

            dual_time = @elapsed begin
                dual_sol = DGFact_dual_solution_from_DDGFact_x(
                    x_cf,
                    F,
                    s,
                    t;
                    l = l,
                    c = c,
                    atol = atol,
                )
            end

            _add_bnb_timing!(:dual_solution_time, dual_time)
            _add_bnb_timing!(:closed_form_uses, 1)

            return _node_result(
                :DDGFact,
                keep,
                x_cf,
                dual_sol.objective_value;
                dual_solution = dual_sol,
                l = l,
                c = c,
                psi = 0.0,
                F = F,
                primal_obj = primal_obj,
                closed_form = true,
            )
        end
    end

    x = Float64[]
    primal_obj = NaN
    F = nothing
    dual_sol = nothing

    bound_time = @elapsed begin
        x, primal_obj, F, dual_sol = _bound_ddgfact_node(
            Ck,
            s,
            t,
            fix1,
            l,
            c;
            atol = atol,
        )
    end

    _add_bnb_timing!(:bound_computation_time, bound_time)

    return _node_result(
        :DDGFact,
        keep,
        x,
        dual_sol.objective_value;
        dual_solution = dual_sol,
        l = l,
        c = c,
        psi = 0.0,
        F = F,
        primal_obj = primal_obj,
    )
end

# Apply DDGFact dual variable fixing.
function _fixing_relaxation(
    ::Val{:DDGFact},
    r,
    s::Int,
    t::Int,
    lb::Float64;
    fixing_rule::Symbol = :dual,
    tol::Float64 = 1e-9,
)
    fixing_rule in (:none, :dual) ||
        error("DDGFact supports only :none and :dual fixing.")

    if fixing_rule == :none
        return (fix_zero = Int[], fix_one = Int[])
    end

    return var_fixing_from_DGFact(
        r.dual_solution.upsilon,
        r.dual_solution.nu,
        r.ub,
        lb;
        l = r.l,
        c = r.c,
        atol = tol,
    )
end
