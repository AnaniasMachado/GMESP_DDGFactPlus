
using CSV
using DataFrames
using Printf

# ============================================================
# Configuration
# ============================================================

data_n = 63
n = 63
t = 2

s_values = [5, 10, 15, 20, 25]

gap_decimals = 3
other_decimals = 1


csv_filepath = joinpath(
    "results/journal_ddgfactplus",
    "bnb_data$(data_n)_n$(n)_t$(t).csv"
)

latex_dir = "results/journal_ddgfactplus/latex_tables"
mkpath(latex_dir)

latex_filepath = joinpath(
    latex_dir,
    "bnb_comparison_n$(n)_t$(t).txt"
)

# ============================================================
# Read CSV
# ============================================================

df = CSV.read(csv_filepath, DataFrame)

required_cols = [
    :s,
    :relaxation,
    :cf_trace,
    :bnb_root_gap,
    :bnb_gap,
    :bnb_nodes,
    :n_fixed_total_bnb,
    :bnb_runtime,
    :closed_form_uses,
    :trace_prunes,
]

missing_cols = setdiff(required_cols, propertynames(df))

isempty(missing_cols) ||
    error("Missing CSV columns: $missing_cols")

# ============================================================
# Formatting
# ============================================================

function fmt_decimal(x, digits)
    ismissing(x) && return "--"
    isfinite(x) || return "--"
    return @sprintf("%.*f", digits, Float64(x))
end

function fmt_integer(x)
    ismissing(x) && return "--"
    return string(round(Int, x))
end

function stat(row, col)
    row === nothing && return "--"

    if col == :bnb_root_gap || col == :bnb_gap
        return fmt_decimal(row[col], gap_decimals)
    elseif col == :bnb_runtime
        return fmt_decimal(row[col], other_decimals)
    else
        return fmt_integer(row[col])
    end
end

# ============================================================
# Find result
# ============================================================

function get_result(df, s, method, cf_trace)
    rows = df[
        (df.s .== s) .&
        (df.relaxation .== method) .&
        (df.cf_trace .== cf_trace),
        :
    ]

    if nrow(rows) == 0
        @warn "Missing result" s method cf_trace
        return nothing
    elseif nrow(rows) > 1
        @warn "Multiple results; using last row" s method cf_trace
    end

    return rows[end, :]
end

# ============================================================
# Write LaTeX table
# ============================================================

open(latex_filepath, "w") do io

    println(io, "\\begin{table}[htbp]")
    println(io, "\\centering")
    println(io, "\\resizebox{\\columnwidth}{!}{%")
    println(io, "\\begin{tabular}{cc|ccccc|cc}")
    println(io, "\\hline")

    # Header
    println(
        io,
        "\$s\$ & Relaxation & Root Gap & Gap & Nodes & Fixed & Time & CF & Prunes \\\\"
    )

    println(io, "\\hline")

    # Five rows for each s
    for s in s_values

        trace = get_result(
            df, s, "Trace", false
        )

        ddg = get_result(
            df, s, "DDGFact", false
        )

        ddg_cf = get_result(
            df, s, "DDGFact_CF", true
        )

        ddgplus = get_result(
            df, s, "DDGFactplus", false
        )

        ddgplus_cf = get_result(
            df, s, "DDGFactplus_CF", true
        )

        # ----------------------------------------------------
        # Trace
        # ----------------------------------------------------

        trace_values = String[
            "\\multirow{5}{*}{$s}",
            "\$\\mathrm{Trace}\$",
            stat(trace, :bnb_root_gap),
            stat(trace, :bnb_gap),
            stat(trace, :bnb_nodes),
            "--",
            stat(trace, :bnb_runtime),
            "--",
            "--",
        ]

        println(io, join(trace_values, " & ") * " \\\\")

        # ----------------------------------------------------
        # DDGFact
        # ----------------------------------------------------

        ddg_values = String[
            "",
            "\$\\mathrm{DDGFact}\$",
            stat(ddg, :bnb_root_gap),
            stat(ddg, :bnb_gap),
            stat(ddg, :bnb_nodes),
            stat(ddg, :n_fixed_total_bnb),
            stat(ddg, :bnb_runtime),
            "--",
            "--",
        ]

        println(io, join(ddg_values, " & ") * " \\\\")

        # ----------------------------------------------------
        # DDGFact (CF)
        # ----------------------------------------------------

        ddg_cf_values = String[
            "",
            "\$\\mathrm{DDGFact}\\text{(CF)}\$",
            stat(ddg_cf, :bnb_root_gap),
            stat(ddg_cf, :bnb_gap),
            stat(ddg_cf, :bnb_nodes),
            stat(ddg_cf, :n_fixed_total_bnb),
            stat(ddg_cf, :bnb_runtime),
            stat(ddg_cf, :closed_form_uses),
            stat(ddg_cf, :trace_prunes),
        ]

        println(io, join(ddg_cf_values, " & ") * " \\\\")

        # ----------------------------------------------------
        # DDGFact+
        # ----------------------------------------------------

        ddgplus_values = String[
            "",
            "\$\\mathrm{DDGFact}^{+}\$",
            stat(ddgplus, :bnb_root_gap),
            stat(ddgplus, :bnb_gap),
            stat(ddgplus, :bnb_nodes),
            stat(ddgplus, :n_fixed_total_bnb),
            stat(ddgplus, :bnb_runtime),
            "--",
            "--",
        ]

        println(io, join(ddgplus_values, " & ") * " \\\\")

        # ----------------------------------------------------
        # DDGFact+ (CF)
        # ----------------------------------------------------

        ddgplus_cf_values = String[
            "",
            "\$\\mathrm{DDGFact}^{+}\\text{(CF)}\$",
            stat(ddgplus_cf, :bnb_root_gap),
            stat(ddgplus_cf, :bnb_gap),
            stat(ddgplus_cf, :bnb_nodes),
            stat(ddgplus_cf, :n_fixed_total_bnb),
            stat(ddgplus_cf, :bnb_runtime),
            stat(ddgplus_cf, :closed_form_uses),
            stat(ddgplus_cf, :trace_prunes),
        ]

        println(io, join(ddgplus_cf_values, " & ") * " \\\\")

        # Separator between s groups
        println(io, "\\hline")
    end

    println(io, "\\end{tabular}")
    println(io, "}")

    println(
        io,
        "\\caption{Branch-and-bound experiments for \$n = $(n)\$ and \$t = $(t)\$.}"
    )

    println(
        io,
        "\\label{tab:bnb_comparison_n$(n)_t$(t)}"
    )

    println(io, "\\end{table}")
end

println("Written: $latex_filepath")
