using JuMP, Gurobi, LinearAlgebra, MosekTools, InvertedIndices

include("utils.jl")

#TODO: warmstart from best incumbent solution
#TODO: warmstarting works poorly when revisiting previous solution

const LAMBDA_TOL = 1e-6

function adaptive_pwl(Q, c; branching = :SOS2, warmstart = true, breakpoint_management = :none, max_breakpoints = 4, large_decrease_threshold = 1/2, custom_termination = true)

    # breakpoint_management in [:aggressive, :standard, :none]
    # branching in [:SOS2, :CC, :MC, :IC, :DCC]

    if !(branching in [:SOS2, :CC, :MC, :IC, :DCC])
        error("Unsupported branching type")
    end
    if !(breakpoint_management in [:aggressive, :standard, :none])
        error("Unsupported breakpoint management strategy")
    end
    if breakpoint_management != :none && !(branching in [:SOS2, :CC, :MC, :IC, :DCC])
        error("Breakpoint management not supported with this branching style")
    end

    n = length(c)

    #Compute eigendecomposition
    eig = eigen(Q)
    B = eig.vectors
    Σ = eig.values

    QP = direct_model(optimizer_with_attributes(
        Gurobi.Optimizer, 
        MOI.Silent() => true,
        "MIPFocus" => 2
    ))

    @variable(QP, x[1:n] >= 0)
    @variable(QP, s_ub[1:n] >= 0)

    @variable(QP, y[1:n])

    @constraint(QP, x .+ s_ub .== 1)

    @constraint(QP, y .== B'*x)

    split_points = [[sum((B[:,i] .<= 0) .* B[:,i]), sum((B[:,i] .>= 0) .* B[:,i])] for i = 1:n]

    if branching in [:SOS2, :CC]
        λ = [[@variable(QP, lower_bound = 0, upper_bound = 1) for j = 1:length(split_points[i])] for i = 1:n]
        
        if branching == :CC
            z = [[@variable(QP, binary = true) for j = 1:(length(split_points[i]) - 1)] for i = 1:n]
        end
    elseif branching in [:DCC]
        λ = [[(@variable(QP, lower_bound = 0, upper_bound = 1), @variable(QP, lower_bound = 0, upper_bound = 1)) for j = 1:(length(split_points[i]) - 1)] for i = 1:n]
        z = [[@variable(QP, binary = true) for j = 1:(length(split_points[i]) - 1)] for i = 1:n]
        @constraint(QP, z_sum[i = 1:n], sum(z[i]) == 1)
        @constraint(QP, [i = 1:n, j = 1:(length(split_points[i]) - 1)], z[i][j] == sum(λ[i][j]))
    elseif branching in [:MC]
        λ = [[@variable(QP) for j = 1:(length(split_points[i])-1)] for i = 1:n]
        z = [[@variable(QP, binary = true) for j = 1:(length(split_points[i])-1)] for i = 1:n]

    elseif branching in [:IC]
        λ = [[@variable(QP, lower_bound = 0, upper_bound = 1) for j = 1:(length(split_points[i])-1)] for i = 1:n]
        z = [[@variable(QP, binary = true) for j = 1:(length(split_points[i])-1)] for i = 1:n]

    end

    iter = 0
    stop_iter = false

    solutions = []
    sol_objs = []

    upper_bound = Inf
    lower_bound = -Inf

    lower_bounds = [lower_bound]
    upper_bounds = [upper_bound]
    times = []

    t_start = time()
    while !stop_iter

        t_iter_start = time()

        sort_order = [sortperm(split_points[i]) for i = 1:n]
        sorted_splits = [split_points[i][sort_order[i]] for i = 1:n]

        if branching in [:SOS2, :CC]
            
            if branching == :SOS2
                @constraint(QP, λ_SOS[i = 1:n], [λ[i][j] for j in sort_order[i]] in SOS2())
                @constraint(QP, λ_y[i = 1:n], sum(λ[i] .* split_points[i]) == y[i])
                @constraint(QP, λ_sum[i = 1:n], sum(λ[i]) == 1)

            elseif branching == :CC
                @constraint(QP, λ_CC[i = 1:n, j = 1:length(split_points[i])], λ[i][sort_order[i][j]] <= (j < length(split_points[i]) ? z[i][j] : 0) + (j > 1 ? z[i][j - 1] : 0))
                @constraint(QP, λ_sum[i = 1:n], sum(λ[i]) == 1)
                @constraint(QP, λ_y[i = 1:n], sum(λ[i] .* split_points[i]) == y[i])
                @constraint(QP, z_sum[i = 1:n], sum(z[i]) == 1)
                
                if warmstart
                    for i = 1:n
                        for j = 1:(length(split_points[i])-1)
                            set_start_value(z[i][j], (sort_order[i][j] == (length(split_points[i])) ? 1 : 0))
                        end
                        if sort_order[i][end] == length(split_points[i])
                            set_start_value(z[i][length(split_points[i])-1], 1)
                        end
                    end

                    for i = 1:n, j = 1:length(split_points[i])
                        set_start_value(λ[i][j], (j == length(split_points[i]) ? 1 : 0))
                    end
                
                end

            end 

            set_objective_sense(QP, MOI.FEASIBILITY_SENSE)
            @objective(QP, Min, sum(sum(Σ[i] * λ[i][j] * split_points[i][j]^2 for j = 1:length(split_points[i])) for i = 1:n) + sum(c[i] * x[i] for i = 1:n))

        elseif branching in [:DCC]

            @constraint(QP, λ_sum[i = 1:n], sum(sorted_splits[i][j] * λ[i][j][1] + sorted_splits[i][j+1] * λ[i][j][2] for j = 1:(length(split_points[i])-1)) == y[i])

            if warmstart
                for i = 1:n
                    for j = 1:(length(split_points[i])-1)
                        set_start_value(z[i][j], sorted_splits[i][j] == split_points[i][end] ? 1 : 0)
                        set_start_value(λ[i][j][1], sorted_splits[i][j] == split_points[i][end] ? 1 : 0)
                    end
                end
            end

            @objective(QP, Min, sum(sum(Σ[i] * (sorted_splits[i][j]^2 * λ[i][j][1] + sorted_splits[i][j+1]^2 * λ[i][j][2]) for j = 1:(length(split_points[i])-1)) for i = 1:n) + sum(c[i] * x[i] for i = 1:n))

        elseif branching in [:MC]

            @constraint(QP, λ_MC_lb[i = 1:n, j = 1:(length(split_points[i])-1)], sorted_splits[i][j] * z[i][j] <= λ[i][j])
            @constraint(QP, λ_MC_ub[i = 1:n, j = 1:(length(split_points[i])-1)], λ[i][j] <= sorted_splits[i][j+1] * z[i][j])

            @constraint(QP, λ_sum[i = 1:n], sum(λ[i]) == y[i])
            @constraint(QP, z_sum[i = 1:n], sum(z[i]) == 1)

            if warmstart
                for i = 1:n
                    for j = 1:(length(split_points[i])-1)
                        set_start_value(z[i][j], sorted_splits[i][j] == split_points[i][end] ? 1 : 0)
                        set_start_value(λ[i][j], sorted_splits[i][j] == split_points[i][end] ? Σ[i] * split_points[i][end]^2 : 0)
                    end
                end
            end

            slope = [[Σ[i] * (sorted_splits[i][j+1]^2 - sorted_splits[i][j]^2)/(sorted_splits[i][j+1] - sorted_splits[i][j]) for j = 1:(length(split_points[i])-1)] for i = 1:n]

            set_objective_sense(QP, MOI.FEASIBILITY_SENSE)
            @objective(QP, Min, sum(sum(Σ[i] * sorted_splits[i][j]^2 * z[i][j] + slope[i][j] * (λ[i][j] - sorted_splits[i][j] * z[i][j]) for j = 1:(length(split_points[i])-1)) for i = 1:n) + sum(c[i] * x[i] for i = 1:n))

        elseif branching in [:IC]

            @constraint(QP, λ_sum[i = 1:n], sum((sorted_splits[i][j+1] - sorted_splits[i][j]) * λ[i][j] for j = 1:(length(split_points[i])-1)) + sorted_splits[i][1] == y[i])

            @constraint(QP, λ_IC_lb[i = 1:n, j = 1:(length(split_points[i])-1)], z[i][j] <= λ[i][j])
            @constraint(QP, λ_IC_ub[i = 1:n, j = 1:(length(split_points[i])-2)], λ[i][j+1] <= z[i][j])

            if warmstart
                for i = 1:n
                    for j = 1:(length(split_points[i])-1)
                        set_start_value(z[i][j], sorted_splits[i][j] <= split_points[i][end] ? 1 : 0)
                        set_start_value(λ[i][j], sorted_splits[i][j] <= split_points[i][end] ? 1 : 0)
                    end
                end
            end

            set_objective_sense(QP, MOI.FEASIBILITY_SENSE)
            @objective(QP, Min, sum(Σ[i] * sorted_splits[i][1]^2 + sum((Σ[i] * (sorted_splits[i][j+1]^2 - sorted_splits[i][j]^2)) * λ[i][j] for j = 1:(length(split_points[i])-1)) for i = 1:n) + sum(c[i] * x[i] for i = 1:n))
            
        end

        global grb_lb = -Inf

        if custom_termination
            function termination_callback(cb_data, cb_where::Cint)

                if cb_where == GRB_CB_MIP

                    objbstP = Ref{Cdouble}()
                    objbndP = Ref{Cdouble}()

                    GRBcbget(cb_data, cb_where, GRB_CB_MIP_OBJBST, objbstP)
                    GRBcbget(cb_data, cb_where, GRB_CB_MIP_OBJBND, objbndP)

                    actual_lb = max(objbndP[], lower_bound)

                    gap = abs((objbstP[] - actual_lb) / objbstP[])
                    
                    if gap < 1e-4
                        println("First terminate")
                        GRBterminate(backend(QP))
                        global grb_lb = actual_lb
                        return
                    end

                    gap_condition = gap < .05

                    ub_improve_condition = objbstP[] < upper_bound - 1e-2

                    if gap_condition && ub_improve_condition
                        GRBterminate(backend(QP))
                        global grb_lb = actual_lb
                        return
                    end
                end
                

            end

            MOI.set(QP, Gurobi.CallbackFunction(), termination_callback)
        end


        set_optimizer_attribute(QP, "Cutoff", upper_bound)
        optimize!(QP)
        objval = objective_value(QP)

        push!(solutions, (value.(y), sum(c[i] * value.(x[i]) for i = 1:n)))
        push!(sol_objs, sum(Σ[i] * value.(y[i])^2 for i = 1:n) + sum(c[i] * value.(x[i]) for i = 1:n))

        upper_bound = minimum(sol_objs)
        # lower_bound = -Inf
        if grb_lb > -Inf
            lower_bound = max(grb_lb, lower_bound)
        else
            lower_bound = max(lower_bound, objval)
        end

        push!(lower_bounds, lower_bound)
        push!(upper_bounds, upper_bound)

        stop_iter = true
        split_vars = []
        for i = 1:n
            if branching in [:SOS2, :CC, :DCC]
                if branching in [:DCC]
                    count_nz = sum(sum(value.(λ[i][j]) .>= 1e-3) for j in 1:(length(split_points[i])-1))
                else
                    count_nz = sum(value.(λ[i]) .>= 1e-3)
                end
                if count_nz > 1
                    stop_iter = false
                    push!(split_vars, (i, value.(y[i])))

                end
            elseif branching in [:MC]
                lbd_diffs = [(value.(λ[i][j]) - sorted_splits[i][j] > 1e-3, sorted_splits[i][j+1] - value.(λ[i][j]) > 1e-3) for j = 1:(length(split_points[i])-1) ]
                if any([all(lbd_diffs[j]) && value.(z[i][j]) > .5 for j = 1:(length(split_points[i])-1)])
                    stop_iter = false
                    push!(split_vars, (i, value.(y[i])))
                end
            elseif branching in [:IC]
                err = min(1 - (sum(value.(λ[i])) % 1), sum(value.(λ[i])) % 1)
                if err > 1e-3 
                    stop_iter = false
                    push!(split_vars, (i, value.(y[i])))
                end
            end
        end

        for (i, val) in split_vars
            if breakpoint_management == :aggressive && length(split_points[i]) == max_breakpoints
                push!(split_points[i], val)
                deleteat!(split_points[i], 3)
                continue
            else
                push!(split_points[i], val)
            end
            if branching in [:SOS2, :CC]
                push!(λ[i], @variable(QP, lower_bound = 0, upper_bound = 1))

                set_normalized_coefficient(λ_sum[i], λ[i][end], 1)

                if branching == :CC
                    push!(z[i], @variable(QP, binary = true))

                    set_normalized_coefficient(λ_y[i], λ[i][end], val)
                    set_normalized_coefficient(z_sum[i], z[i][end], 1)
                end

            elseif branching in [:DCC]
                push!(λ[i], (@variable(QP, lower_bound = 0, upper_bound = 1), @variable(QP, lower_bound = 0, upper_bound = 1)))
                push!(z[i], @variable(QP, binary = true))

                @constraint(QP, z[i][end] == sum(λ[i][end]))
                set_normalized_coefficient(z_sum[i], z[i][end], 1)

            elseif branching in [:MC]

                push!(λ[i], @variable(QP))
                push!(z[i], @variable(QP, binary = true))

                set_normalized_coefficient(λ_sum[i], λ[i][end], 1)
                set_normalized_coefficient(z_sum[i], z[i][end], 1)

            elseif branching in [:IC]
                push!(λ[i], @variable(QP, lower_bound = 0, upper_bound = 1))
                push!(z[i], @variable(QP, binary = true))
            end
        end

        if branching in [:SOS2]
            delete.(QP, λ_SOS)
            unregister.(QP, :λ_SOS)
            delete.(QP, λ_y)
            unregister.(QP, :λ_y)
            delete.(QP, λ_sum)
            unregister.(QP, :λ_sum)
        elseif branching in [:CC]
            delete.(QP, λ_CC)
            unregister.(QP, :λ_CC)
            delete.(QP, λ_sum)
            unregister.(QP, :λ_sum)
            delete.(QP, z_sum)
            unregister.(QP, :z_sum)
            delete.(QP, λ_y)
            unregister.(QP, :λ_y)
        elseif branching in [:MC]
            delete.(QP, λ_MC_lb)
            unregister.(QP, :λ_MC_lb)
            delete.(QP, λ_MC_ub)
            unregister.(QP, :λ_MC_ub)
            delete.(QP, λ_sum)
            unregister.(QP, :λ_sum)
            delete.(QP, z_sum)
            unregister.(QP, :z_sum)
        elseif branching in [:IC]
            delete.(QP, λ_sum)
            unregister.(QP, :λ_sum)
            delete.(QP, λ_IC_lb)
            unregister.(QP, :λ_IC_lb)
            delete.(QP, λ_IC_ub)
            unregister.(QP, :λ_IC_ub)
        elseif branching in [:DCC]
            delete.(QP, λ_sum)
            unregister.(QP, :λ_sum)
        end

        
        sort_order = [sortperm(split_points[i]) for i = 1:n]
        sorted_splits = [split_points[i][sort_order[i]] for i = 1:n]

        if breakpoint_management == :standard
            check_delete = true
            while check_delete
                check_delete = false
                for i = n:-1:1
                    for j = 2:(length(sorted_splits[i])-1)
                        delete_current = true
                        for k = 1:length(solutions)
                            orig_obj = sum(evaluate_pwl(sorted_splits[t], Σ[t] * sorted_splits[t].^2, solutions[k][1][t]) for t = 1:n) + solutions[k][2]
                            obj_eval = sum(evaluate_pwl(sorted_splits[t][t == i ? Not(j) : 1:end], Σ[t] * sorted_splits[t][t == i ? Not(j) : 1:end].^2, solutions[k][1][t]) for t = 1:n) + solutions[k][2]
                            is_large_decrease = obj_eval < (upper_bound + large_decrease_threshold * (sol_objs[k] - upper_bound))
                            if (obj_eval < upper_bound + 1e-3 && obj_eval + 1e-6 < orig_obj) || is_large_decrease
                                delete_current = false
                                break
                            end
                        end
                        if delete_current
                            deleteat!(split_points[i], findfirst(==(sorted_splits[i][j]), split_points[i]))
                            if branching in [:SOS2, :CC]
                                delete.(QP, λ[i][end])
                                deleteat!(λ[i], length(λ[i]))
                                if branching == :CC
                                    delete.(QP, z[i][end])
                                    deleteat!(z[i], length(z[i]))
                                end
                            elseif branching in [:MC, :IC]
                                delete.(QP, λ[i][end])
                                deleteat!(λ[i], length(λ[i]))
                                delete.(QP, z[i][end])
                                deleteat!(z[i], length(z[i]))
                            end

                            check_delete = true
                            println("Deleted: ", (iter,i,j))

                            sort_order = [sortperm(split_points[i]) for i = 1:n]
                            sorted_splits = [split_points[i][sort_order[i]] for i = 1:n]
                            break
                        end
                    end
                end
            end

        end


        iter +=1 
        t_iter_end = time()

        push!(times, t_iter_end - t_iter_start)

        println(repeat("=", 40))
        println("Iter: ", iter)
        println("Num Updates: ", length(split_vars))
        println("Objective: ", objval)
        println("Lower Bound: ", lower_bound)
        println("Upper Bound: ", upper_bound)
        println("Time: ", round(time() - t_start, digits = 2))
        println("Iter Time: ", t_iter_end - t_iter_start)

    end

    tot_time = time() - t_start

    return iter, tot_time, times, lower_bounds, upper_bounds

end

function spatial_branch(Q, c; branching_n = Inf)

    n = length(c)

    #Compute eigendecomposition
    eig = eigen(Q)
    B = eig.vectors
    Σ = eig.values

    QP = Model(optimizer_with_attributes(
        Gurobi.Optimizer, 
        MOI.Silent() => true
    ))

    @variable(QP, x[1:n] >= 0)
    @variable(QP, s_ub[1:n] >= 0)

    @variable(QP, y[1:n])

    @constraint(QP, x .+ s_ub .== 1)

    @constraint(QP, y .== B'*x)

    @objective(QP, Min, sum(c[i] * x[i] for i = 1:n))

    initial_bounds = [[sum((B[:,i] .<= 0) .* B[:,i]), sum((B[:,i] .>= 0) .* B[:,i])] for i = 1:n]

    open_model_bounds = [ (initial_bounds, -Inf, Inf, 1) ] #(bound_dictionary, lower_bound, upper_bound, level)
    global_ub = Inf
    global_lb = -Inf

    incumbent_bounds = nothing
    incumbent_sol = nothing

    iter = 0

    lower_bounds = []
    upper_bounds = []
    times = []

    t_start = time()

    solved = true
    while true

        if time() - t_start > 30*60
            solved = false
            break
        end

        t_iter_start = time()

        #Select next subproblem
        selected_subproblem_index = nothing
        for i in eachindex(open_model_bounds)
            if open_model_bounds[i][2] == global_lb
                selected_subproblem_index = i
            end
        end
        
        #Get selected subproblem and bounds
        (subp_bounds, subp_lb, subp_ub, subp_level) = open_model_bounds[selected_subproblem_index]
        deleteat!(open_model_bounds, selected_subproblem_index)

        #Update model with subproblem bounds and linear objective
        obj_const = 0
        for i = 1:n

            set_lower_bound(y[i], subp_bounds[i][1])
            set_upper_bound(y[i], subp_bounds[i][2])

            lb_obj = Σ[i] * subp_bounds[i][1]^2
            ub_obj = Σ[i] * subp_bounds[i][2]^2
            obj_coef = (ub_obj - lb_obj) / (subp_bounds[i][2] - subp_bounds[i][1])

            set_objective_coefficient(QP, y[i], obj_coef)
            obj_const += lb_obj - obj_coef * subp_bounds[i][1]

        end

        optimize!( QP )
        if termination_status(QP) == INFEASIBLE
            subp_lb = Inf
            subp_ub = Inf
            println("Region Infeasible")

        elseif termination_status(QP) == OPTIMAL

            subp_obj_val = objective_value( QP )
            subp_lb = subp_obj_val + obj_const
            subp_ub = sum(Σ[i] * value.(y[i])^2 for i = 1:n) + sum(c[i]*value.(x[i]) for i = 1:n)

            best_sol_in_region = (subp_ub <= global_ub)

            #Update upper bound
            if subp_ub < global_ub
                incumbent_bounds = subp_bounds
                incumbent_sol = value.(y)
                global_ub = subp_ub
            end

        else
            throw(error("Unhandled termination status: ", termination_status(BattOpt)))
        end

        #Update lower bound
        global_lb = subp_lb
        for i in eachindex(open_model_bounds)
            if open_model_bounds[i][2] < global_lb
                global_lb = open_model_bounds[i][2]
            end
        end

        if round(global_lb, digits = 2) == round(global_ub, digits = 2)
            println("Iter: ", iter, "    ================================================")
            println("Level: ", subp_level)
            println("Upper Bound: ", global_ub)
            println("Remaining problems: ", length(open_model_bounds))
            println("================================================================")
            break
        end

        #Fathom based on new upper bound
        open_model_bounds = filter(s -> (1 + 1e-5) * s[2] < global_ub, open_model_bounds)

        #Branch
        if subp_lb < global_ub
            if subp_level % branching_n == 0
                longest_edge_length = -Inf
                longest_edge_var = nothing
                for i = 1:n
                    if subp_bounds[i][2] - subp_bounds[i][1] > longest_edge_length
                        longest_edge_var = i
                        longest_edge_length = subp_bounds[i][2] - subp_bounds[i][1]
                    end
                end
                split_var = longest_edge_var
                split_point = (subp_bounds[split_var][2] + subp_bounds[split_var][1]) / 2

            else

                worst_edge_err = -Inf
                worst_edge_var = nothing

                for i = 1:n

                    lb_obj = Σ[i] * subp_bounds[i][1]^2
                    ub_obj = Σ[i] * subp_bounds[i][2]^2
                    obj_coef_var = (ub_obj - lb_obj) / (subp_bounds[i][2] - subp_bounds[i][1])
                    obj_const_var = lb_obj - obj_coef_var * subp_bounds[i][1]

                    edge_err = (Σ[i] * value.(y[i])^2) - (obj_coef_var * value(y[i]) + obj_const_var)

                    if edge_err > worst_edge_err
                        worst_edge_var = i
                        worst_edge_err = edge_err
                    end

                end

                split_var = worst_edge_var

                if best_sol_in_region && value.(y[split_var]) > subp_bounds[split_var][1] && value.(y[split_var]) < subp_bounds[split_var][2]
                    split_point = value.(y[split_var])
                else
                    split_point = (subp_bounds[split_var][2] + subp_bounds[split_var][1]) / 2
                end
                
            end

            partition_lower_bounds = deepcopy(subp_bounds)
            partition_upper_bounds = deepcopy(subp_bounds)

            partition_lower_bounds[split_var] = [subp_bounds[split_var][1], split_point]
            partition_upper_bounds[split_var] = [split_point, subp_bounds[split_var][2]]

            push!(open_model_bounds, (partition_lower_bounds, subp_lb, subp_ub, subp_level + 1))
            push!(open_model_bounds, (partition_upper_bounds, subp_lb, subp_ub, subp_level + 1))

        end

        iter += 1

        t_iter_end = time()

        push!(times, t_iter_end - t_iter_start)
        push!(lower_bounds, global_lb)
        push!(upper_bounds, global_ub)

        println("Iter: ", iter, "    ================================================")
        println("Level: ", subp_level)
        println("Upper Bound: ", global_ub)
        println("Lower Bound: ", global_lb)
        println("Remaining problems: ", length(open_model_bounds))
        println("================================================================")

    end

    if solved
        tot_time = time() - t_start
    else
        tot_time = Inf
    end

    return iter, tot_time, times, lower_bounds, upper_bounds

end

function kkt_branch(Q, c)

    n = length(c)

    Q = 2 .* Q

    Qext = zeros(n+1,n+1)
    Qext[2:end,2:end] = 1/2 .* Q
    Qext[2:end,1] = 1/2 .* c
    Qext[1,2:end] = 1/2 .* c

    SDP = Model(optimizer_with_attributes(
        Mosek.Optimizer,
        MOI.Silent() => true
    ))

    @variable(SDP, Y[1:(n+1),1:(n+1)] in PSDCone())
    @variable(SDP, x[1:n], lower_bound = 0, upper_bound = 1)
    @variable(SDP, y[1:n], lower_bound = 0)
    @variable(SDP, z[1:n], lower_bound = 0)

    @constraint(SDP, y .- z .== -Q*x .- c)
    @constraint(SDP, Y[1:end,1] .== [1; x])
    @constraint(SDP, [i = 2:n+1], 0 .<= Y[2:end,i])
    @constraint(SDP, [i = 2:n+1], Y[2:end,i] .<= Y[1,i])

    @constraint(SDP, sum(Qext .* Y) == (-sum(y) + sum(c[i] * x[i] for i = 1:n))/2)

    @objective(SDP, Min, sum(Qext .* Y))

    subproblems = [(-Inf, [],[],[],[])] #(lb, F0, F1, Fy, Fz)

    global_ub = Inf
    global_lb = -Inf

    incumbent_sol = nothing

    iter = 0

    lower_bounds = []
    upper_bounds = []
    times = []

    t_start = time()

    while true

        t_iter_start = time()

        selected_subproblem_index = nothing
        for i in eachindex(subproblems)
            if subproblems[i][1] == global_lb
                selected_subproblem_index = i
            end
        end

        (subp_lb, F0, F1, Fy, Fz) = subproblems[selected_subproblem_index]
        deleteat!(subproblems, selected_subproblem_index)

        for i = 1:n
            if i in F0
                fix(x[i], 0, force = true)
            elseif i in F1
                fix(x[i], 1, force = true)
            elseif is_fixed(x[i])
                unfix(x[i])
                set_lower_bound(x[i], 0)
                set_upper_bound(x[i], 1)
            end

            if i in Fy
                fix(y[i], 0, force = true)
            elseif is_fixed(y[i])
                unfix(y[i])
                set_lower_bound(y[i], 0)
            end

            if i in Fz
                fix(z[i], 0, force = true)
            elseif is_fixed(z[i])
                unfix(z[i])
                set_lower_bound(z[i], 0)
            end
        end

        optimize!(SDP)
        objval = objective_value(SDP)

        satisfies_complementarity = true
        max_viol_val = -Inf
        max_viol_idx = nothing
        for i = 1:n
            if value.(x[i]) * value.(z[i]) > 1e-6 && !(i in union(F1, Fy))
                satisfies_complementarity = false
                viol = value.(x[i]) * value.(z[i]) / (-c[i] - sum(Q[i,:] .* (Q[i,:] .< 0)))
                if viol > max_viol_val
                    max_viol_val = viol
                    max_viol_idx = (i, :z)
                end
            end
            if (1 - value.(x[i])) * value.(y[i]) > 1e-6 && !(i in union(F0, Fz))
                satisfies_complementarity = false
                viol = (1 - value.(x[i])) * value.(y[i]) / (c[i] + sum(Q[i,:] .* (Q[i,:] .> 0)))
                if viol > max_viol_val
                    max_viol_val = viol
                    max_viol_idx = (i, :y)
                end
            end
        end

        #Update lower bound
        global_lb = objval
        for i in eachindex(subproblems)
            if subproblems[i][1] < global_lb
                global_lb = subproblems[i][1]
            end
        end

        #Update upper bound
        if satisfies_complementarity && objval < global_ub

            incumbent_sol = value.(x)
            global_ub = objval
        end

        if round(global_lb, digits = 2) >= round(global_ub, digits = 2)
            println("Iter: ", iter, "    ================================================")
            println("Upper Bound: ", global_ub)
            println("Remaining problems: ", length(subproblems))
            println("================================================================")
            break
        end

        #Fathom based on new upper bound
        subproblems = filter(s -> (1 + 1e-5) * s[1] < global_ub, subproblems)

        if !satisfies_complementarity

            if max_viol_idx[2] == :y
                #case (ii)
                new_F0 = copy(F0)
                push!(new_F0, max_viol_idx[1])

                new_Fy = copy(Fy)
                push!(new_Fy, max_viol_idx[1])

                new_Fz = copy(Fz)
                push!(new_Fz, max_viol_idx[1])

                new_case_1 = (objval, new_F0, copy(F1), new_Fy, copy(Fz))
                new_case_2 = (objval, copy(F0), copy(F1), copy(Fy), new_Fz)

            elseif max_viol_idx[2] == :z
                #case (i)
                new_F1 = copy(F1)
                push!(new_F1, max_viol_idx[1])

                new_Fy = copy(Fy)
                push!(new_Fy, max_viol_idx[1])

                new_Fz = copy(Fz)
                push!(new_Fz, max_viol_idx[1])

                new_case_1 = (objval, copy(F0), new_F1, copy(Fy), new_Fz)
                new_case_2 = (objval, copy(F0), copy(F1), new_Fy, copy(Fz))

            end

            push!(subproblems, new_case_1)
            push!(subproblems, new_case_2)

        end

        iter += 1

        t_iter_end = time()

        push!(times, t_iter_end - t_iter_start)
        push!(lower_bounds, global_lb)
        push!(upper_bounds, global_ub)

        println("Iter: ", iter, "    ================================================")
        println("Upper Bound: ", global_ub)
        println("Lower Bound: ", global_lb)
        println("Remaining problems: ", length(subproblems))
        println("================================================================")

    end

    tot_time = time() - t_start

    return iter, tot_time, times, lower_bounds, upper_bounds


end