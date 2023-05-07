using CSV, DataFrames

include("solve.jl")

function run_branching_test(n, d, s, b)

    csv_data = DataFrame(
        n = Int64[], d = Float64[], s = Int64[], b = String[], iter = Int64[], time = Float64[]
    )

    Q, c = build_box_concave_quadratic(n, d, seed = s, type = :negative)

    iter, tot_time, times, lower_bounds, upper_bounds = adaptive_pwl(
        Q, c, branching = b, warmstart = true, breakpoint_management = :standard, large_decrease_threshold = 2/3, custom_termination = false
    )

    push!(csv_data, [n, d, s, string(b), iter, tot_time])

    return csv_data

end

ns = 5:5:20
d = 1
ss = [1, 2, 3, 4, 5]
branchings = [:SOS2, :CC, :MC, :DCC, :IC]

data_file = "results/run_branching"

if !isfile(string(data_file, ".csv"))
    CSV.write(string(data_file, ".csv"), [], writeheader=true, header=[
        "n", "d", "s", "Branching", "Iter", "Time"
        ])
end

for n in ns, s in ss, b in branchings
    branching_csv = run_branching_test(n, d, s, b)
    CSV.write(string(data_file, ".csv"), branching_csv, decimal = '.', append = true, delim = ",")

end




