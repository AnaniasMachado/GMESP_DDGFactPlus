using CSV
using DataFrames
using Plots
using LaTeXStrings
using Statistics

n_values = [63, 90]
kappa_values = [2, 4, 6, 8]

function gen_runtime_plot(n, kappa_values)
    csv_dir = "results/journal/csv_tables"
    plots_dir = "plots/journal"

    mkpath(plots_dir)

    dfs = DataFrame[]

    for kappa in kappa_values
        results_filepath = joinpath(
            csv_dir,
            "results_root_gap_n$(n)_kappa$(kappa).csv"
        )

        df = CSV.read(results_filepath, DataFrame)
        df.kappa .= kappa

        push!(dfs, df)
    end

    df_all = vcat(dfs...)

    df_avg = combine(
        groupby(df_all, :s),
        :ddgfact_runtime => mean => :ddgfact_runtime,
        :ddgfact_plus_runtime => mean => :ddgfact_plus_runtime,
        :glinx_runtime => mean => :glinx_runtime,
    )

    sort!(df_avg, :s)

    default(
        linewidth=2,
        markersize=5,
        marker=:circle,
        markerstrokewidth=0,
        framestyle=:box,
        grid=true,
        size=(700, 500),
        dpi=300,
    )

    plt = plot(
        df_avg.s,
        df_avg.ddgfact_runtime;
        label=L"\mathrm{DDGFact}",
        xlabel=L"s",
        ylabel="Average runtime (sec)",
        # title=L"n = %$(n)",
        title="",
        yscale=:log10,
    )

    plot!(
        plt,
        df_avg.s,
        df_avg.ddgfact_plus_runtime;
        label=L"\mathrm{DDGFact}^{+}",
    )

    plot!(
        plt,
        df_avg.s,
        df_avg.glinx_runtime;
        label=L"\mathrm{glinx}_{\gamma}",
    )

    output_filepath = joinpath(
        plots_dir,
        "average_runtimes_n$(n).png"
    )

    savefig(plt, output_filepath)

    println("Saved plot to: $output_filepath")
end

for n in n_values
    gen_runtime_plot(n, kappa_values)
end