using CSV
using DataFrames
using Statistics
using Printf

n_values = [63, 90]
kappa_values = [2, 4, 6, 8]

function gen_runtime_table(n, kappa_values)
    csv_dir = "results/journal_ddgfactplus/csv_tables"
    output_dir = "results/journal_ddgfactplus/latex_tables"

    mkpath(output_dir)

    dfs = DataFrame[]

    for kappa in kappa_values
        results_filepath = joinpath(
            csv_dir,
            "results_root_gap_n$(n)_kappa$(kappa).csv"
        )

        df = CSV.read(results_filepath, DataFrame)
        push!(dfs, df)
    end

    df_all = vcat(dfs...)

    methods = [
        ("DDGFact", :ddgfact_runtime),
        ("DDGFact+", :ddgfact_plus_runtime),
        ("Glinx", :glinx_runtime),
    ]

    output_filepath = joinpath(
        output_dir,
        "runtime_stats_n$(n).txt"
    )

    open(output_filepath, "w") do io
        println(io, raw"\begin{table}[htbp]")
        println(io, raw"\centering")
        println(io, raw"\caption{Runtime statistics for $n = " * string(n) * raw"$.}")
        println(io, raw"\label{tab:runtime_stats_n" * string(n) * "}")
        println(io, raw"\begin{tabular}{llrrrr}")
        println(io, raw"\toprule")
        println(io, "\$s\$ & Method & Min & Mean & Max & Std. Dev. \\\\")
        println(io, raw"\midrule")

        grouped = groupby(sort(df_all, :s), :s)

        for (i, group) in enumerate(grouped)
            s = first(group.s)

            for (method_name, runtime_col) in methods
                values = collect(skipmissing(group[!, runtime_col]))

                min_val = minimum(values)
                mean_val = mean(values)
                max_val = maximum(values)
                std_val = length(values) > 1 ? std(values) : NaN

                @printf(
                    io,
                    "%s & %s & %.4f & %.4f & %.4f & %.4f \\\\\n",
                    string(s),
                    method_name,
                    min_val,
                    mean_val,
                    max_val,
                    std_val
                )
            end

            if i < length(grouped)
                println(io, raw"\midrule")
            end
        end

        println(io, raw"\bottomrule")
        println(io, raw"\end{tabular}")
        println(io, raw"\end{table}")
    end

    println("Saved LaTeX table to: $output_filepath")
end

for n in n_values
    gen_runtime_table(n, kappa_values)
end