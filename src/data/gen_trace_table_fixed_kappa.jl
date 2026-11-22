
using CSV
using DataFrames
using Printf

# ============================================================
# Configuration
# ============================================================

data_n = 63
n = 63
kappa = 5

s_values = [10, 20, 30, 40, 50]

gap_decimals = 3
other_decimals = 1

csv_filepath = joinpath(
    "results/journal_ddgfactplus",
    "bnb_data$(data_n)_n$(n)_kappa$(kappa).csv"
)

latex_dir = "results/journal_ddgfactplus/latex_tables"
mkpath(latex_dir)

latex_filepath = joinpath(
    latex_dir,
    "bnb_comparison_n$(n)_kappa$(kappa).txt"
)

# ============================================================
# Read CSV
# ============================================================

df = CSV.read(csv_filepath, DataFrame)

required_cols = [
    :s,
    :t,
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

function get_result(df, s, t, method, cf_trace)
    rows = df[
        (df.s .== s) .&
        (df.t .== t) .&
        (df.relaxation .== method) .&
        (df.cf_trace .== cf_trace),
        :
    ]

    if nrow(rows) == 0
        @warn "Missing result" s t method cf_trace
        return nothing
    elseif nrow(rows) > 1
        @warn "Multiple results; using last row" s t method cf_trace
    end

    return rows[end, :]
end

# ============================================================
# Collect results
# ============================================================

methods = [
    ("Trace", false, "\$\\mathrm{Trace}\$"),
    ("DDGFact", false, "\$\\mathrm{DDGFact}\$"),
    ("DDGFact_CF", true, "\$\\mathrm{DDGFact}\\text{(CF)}\$"),
    ("DDGFactplus", false, "\$\\mathrm{DDGFact}^{+}\$"),
    ("DDGFactplus_CF", true, "\$\\mathrm{DDGFact}^{+}\\text{(CF)}\$"),
]

results = []

for s in s_values

    t = s - kappa

    1 <= t <= s ||
        error("Invalid parameters: s=$s, t=$t, kappa=$kappa")

    for (method, cf_trace, latex_name) in methods

        r = get_result(df, s, t, method, cf_trace)

        push!(
            results,
            (
                s = s,
                t = t,
                method = method,
                latex_name = latex_name,
                row = r,
            )
        )
    end
end

# ============================================================
# Determine optional columns
# ============================================================

function has_nonzero(results, col)
    return any(results) do result
        row = result.row

        row === nothing && return false

        value = row[col]

        return !ismissing(value) && value != 0
    end
end

show_cf = has_nonzero(results, :closed_form_uses)
show_prunes = has_nonzero(results, :trace_prunes)

println("Include CF column: ", show_cf)
println("Include Prunes column: ", show_prunes)

# ============================================================
# Write LaTeX table
# ============================================================

open(latex_filepath, "w") do io

    println(io, "\\begin{table}[htbp]")
    println(io, "\\centering")
    println(io, "\\resizebox{\\columnwidth}{!}{%")

    # Dynamic column specification
    # s, t, Relaxation | Root Gap, Gap, Nodes, Fixed, Time | CF, Prunes
    colspec = "ccc|ccccc"

    if show_cf || show_prunes
        colspec *= "|"
        show_cf && (colspec *= "c")
        show_prunes && (colspec *= "c")
    end

    println(io, "\\begin{tabular}{$colspec}")
    println(io, "\\hline")

    # Dynamic header
    headers = String[
        "\$s\$",
        "\$t\$",
        "Relaxation",
        "Root Gap",
        "Gap",
        "Nodes",
        "Fixed",
        "Time",
    ]

    show_cf && push!(headers, "CF")
    show_prunes && push!(headers, "Prunes")

    println(io, join(headers, " & ") * " \\\\")
    println(io, "\\hline")

    # Five rows per (s,t) group
    for s in s_values

        t = s - kappa

        group_results = filter(
            result -> result.s == s && result.t == t,
            results
        )

        for (i, result) in enumerate(group_results)

            r = result.row

            values = String[
                i == 1 ? "\\multirow{5}{*}{$s}" : "",
                i == 1 ? "\\multirow{5}{*}{$t}" : "",
                result.latex_name,
                stat(r, :bnb_root_gap),
                stat(r, :bnb_gap),
                stat(r, :bnb_nodes),
                result.method == "Trace" ?
                    "--" : stat(r, :n_fixed_total_bnb),
                stat(r, :bnb_runtime),
            ]

            if show_cf
                push!(
                    values,
                    endswith(result.method, "_CF") ?
                        stat(r, :closed_form_uses) : "--"
                )
            end

            if show_prunes
                push!(
                    values,
                    endswith(result.method, "_CF") ?
                        stat(r, :trace_prunes) : "--"
                )
            end

            println(io, join(values, " & ") * " \\\\")
        end

        println(io, "\\hline")
    end

    println(io, "\\end{tabular}")
    println(io, "}")

    println(
        io,
        "\\caption{Branch-and-bound experiments for \$n = $(n)\$ and \$t = s - \\kappa\$, with \$\\kappa = $(kappa)\$.}"
    )

    println(
        io,
        "\\label{tab:bnb_comparison_n$(n)_kappa$(kappa)}"
    )

    println(io, "\\end{table}")
end

println("Written: $latex_filepath")
