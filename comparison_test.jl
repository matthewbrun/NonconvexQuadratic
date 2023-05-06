using CSV, DataFrames

include("solve.jl")

function run_comparison_test(n, d, s)

    csv_data = DataFrame(
        n = Int64[], d = Float64[], s = Int64[], apwl_iter = Int64[], sb_iter = Int64[], kb_iter = Int64[], apwl_time = Float64[], sb_time = Float64[], kb_time = Float64[]
    )

    Q, c = build_box_concave_quadratic(n, d, seed = s, type = :negative)

    iter_apwl, tot_time_apwl, times, lower_bounds, upper_bounds = adaptive_pwl(
        Q, c, branching = :MC, warmstart = true, breakpoint_management = :standard, large_decrease_threshold = 2/3, custom_termination = true
    )

    iter_sb, tot_time_sb, times, lower_bounds, upper_bounds = spatial_branch(
        Q, c
    )

    iter_kb, tot_time_kb, times, lower_bounds, upper_bounds = kkt_branch(
        Q, c
    )

    push!(csv_data, [n, d, s, iter_apwl, iter_sb, iter_kb, tot_time_apwl, tot_time_sb, tot_time_kb])

    return csv_data

end



ns = 5:5:20
d = 1
ss = [1, 2, 3, 4, 5]

data_file = "results/run_comparisons"

if !isfile(string(data_file, ".csv"))
    CSV.write(string(data_file, ".csv"), [], writeheader=true, header=[
        "n", "d", "s", "Adaptive PWL Iter", "Spatial Branch Iter", "KKT Branch Iter", "Adaptive PWL Time", "Spatial Branch Time", "KKT Branch Time"
        ])
end

for n in ns, s in ss
    compare_csv = run_comparison_test(n, d, s)
    CSV.write(string(data_file, ".csv"), compare_csv, decimal = '.', append = true, delim = ",")

end




