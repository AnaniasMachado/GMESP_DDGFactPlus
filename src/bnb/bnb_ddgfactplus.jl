
using LinearAlgebra

# Solve the DDGFact+ relaxation and construct its dual solution.
function _bound_ddgfactplus_node(
    Ck::Symmetric,
    s::Int,
    t::Int,
    fix1::Vector{Int},
    l::Vector{Float64},
    c::Vector{Float64};
    psi::Union{Nothing,Float64},
    atol::Float64,
    psi_margin::Float64,
    psi_floor::Float64,
)
    psi_node =
        psi === nothing ?
        _default_psi(
            Ck;
            psi_margin = psi_margin,
            psi_floor = psi_floor,
        ) :
        psi

    x = Float64[]
    primal_obj = NaN

    relaxation_time = @elapsed begin
        x, primal_obj = aug_ddfact_gmesp(
            Ck,
            s,
            t,
            psi_node;
            J1 = fix1,
            atol = atol,
        )
    end

    _add_bnb_timing!(:relaxation_solve_time, relaxation_time)
    _add_bnb_timing!(:knitro_time, relaxation_time)

    F = nothing

    factorization_time = @elapsed begin
        F = compute_F(
            Ck;
            psi = psi_node,
            atol = atol,
        )
    end

    _add_bnb_timing!(:factorization_time, factorization_time)

    dual_sol = nothing

    dual_time = @elapsed begin
        dual_sol = DGFactplus_dual_solution_from_DDGFactplus_x(
            x,
            F,
            s,
            t,
            psi_node;
            l = l,
            c = c,
            atol = atol,
        )
    end

    _add_bnb_timing!(:dual_solution_time, dual_time)

    return x, primal_obj, F, dual_sol, psi_node
end

# Bound a node using DDGFact+.
function _bound_relaxation_node(
    ::Val{:DDGFactplus},
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
    psi_node =
        psi === nothing ?
        _default_psi(
            Ck;
            psi_margin = psi_margin,
            psi_floor = psi_floor,
        ) :
        psi

    if cf_trace
        valid = false
        x_cf = Float64[]

        cf_time = @elapsed begin
            valid, x_cf, _ = closed_form_general(
                Ck,
                s,
                t,
                psi_node;
                J1 = fix1,
                atol = atol,
            )
        end

        _add_bnb_timing!(:closed_form_time, cf_time)

        if valid
            primal_obj = DDGFactplus_value_at_x(
                x_cf,
                Ck,
                t,
                psi_node;
                atol = atol,
            )

            F = nothing

            factorization_time = @elapsed begin
                F = compute_F(
                    Ck;
                    psi = psi_node,
                    atol = atol,
                )
            end

            _add_bnb_timing!(
                :factorization_time,
                factorization_time,
            )

            dual_sol = nothing

            dual_time = @elapsed begin
                dual_sol =
                    DGFactplus_dual_solution_from_DDGFactplus_x(
                        x_cf,
                        F,
                        s,
                        t,
                        psi_node;
                        l = l,
                        c = c,
                        atol = atol,
                    )
            end

            _add_bnb_timing!(:dual_solution_time, dual_time)
            _add_bnb_timing!(:closed_form_uses, 1)

            return _node_result(
                :DDGFactplus,
                keep,
                x_cf,
                dual_sol.objective_value;
                dual_solution = dual_sol,
                l = l,
                c = c,
                psi = psi_node,
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
        x, primal_obj, F, dual_sol, psi_node =
            _bound_ddgfactplus_node(
                Ck,
                s,
                t,
                fix1,
                l,
                c;
                psi = psi_node,
                atol = atol,
                psi_margin = psi_margin,
                psi_floor = psi_floor,
            )
    end

    _add_bnb_timing!(:bound_computation_time, bound_time)

    return _node_result(
        :DDGFactplus,
        keep,
        x,
        dual_sol.objective_value;
        dual_solution = dual_sol,
        l = l,
        c = c,
        psi = psi_node,
        F = F,
        primal_obj = primal_obj,
        closed_form = false,
    )
end

# Apply DDGFact+ primal and dual variable fixing.
function _fixing_relaxation(
    ::Val{:DDGFactplus},
    r,
    s::Int,
    t::Int,
    lb::Float64;
    fixing_rule::Symbol = :dual,
    tol::Float64 = 1e-9,
)
    fixing_rule = _normalize_fixing_rule(fixing_rule)

    fix_zero = Int[]
    fix_one = Int[]

    if fixing_rule in (:dual, :both)
        fixing_dual = var_fixing_from_DGFactplus(
            r.dual_solution.upsilon,
            r.dual_solution.nu,
            r.ub,
            lb;
            l = r.l,
            c = r.c,
            atol = tol,
        )

        append!(fix_zero, fixing_dual.fix_zero)
        append!(fix_one, fixing_dual.fix_one)
    end

    if fixing_rule in (:primal, :both)
        fixing_primal = var_fixing_DDGFactplus_primal(
            r.x,
            r.F,
            s,
            t,
            r.psi,
            lb;
            l = r.l,
            c = r.c,
            atol = tol,
        )

        append!(fix_zero, fixing_primal.fix_zero)
        append!(fix_one, fixing_primal.fix_one)
    end

    return (
        fix_zero = sort(unique(fix_zero)),
        fix_one = sort(unique(fix_one)),
    )
end
