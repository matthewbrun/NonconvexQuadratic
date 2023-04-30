using JuMP, Gurobi, LinearAlgebra, MosekTools

include("utils.jl")

const LAMBDA_TOL = 1e-6

function adaptive_pwl(Q, c)

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

    split_points = [[sum((B[:,i] .<= 0) .* B[:,i]), sum((B[:,i] .>= 0) .* B[:,i])] for i = 1:n]

    λ = [[@variable(QP, lower_bound = 0, upper_bound = 1) for j = 1:length(split_points[i])] for i = 1:n]
    @constraint(QP, λ_sum[i = 1:n], sum(λ[i]) == 1)
    @constraint(QP, λ_y[i = 1:n], sum(λ[i] .* split_points[i]) == y[i])

    iter = 0
    stop_iter = false

    t_start = time()
    while !stop_iter

        sort_order = [sortperm(split_points[i]) for i = 1:n]
        @constraint(QP, λ_SOS[i = 1:n], [λ[i][j] for j in sort_order[i]] in SOS2())

        @objective(QP, Min, sum(sum(Σ[i] * λ[i][j] * split_points[i][j]^2 for j = 1:length(split_points[i])) for i = 1:n) + sum(c[i] * x[i] for i = 1:n))

        optimize!(QP)
        objval = objective_value(QP)

        stop_iter = true
        split_vars = []
        for i = 1:n
            count_nz = sum(value.(λ[i]) .>= 1e-3)
            if count_nz > 1
                stop_iter = false
                push!(split_vars, (i, sum(value.(λ[i]) .* split_points[i])))

            end
        end

        for (i, val) in split_vars
            push!(split_points[i], val)
            push!(λ[i], @variable(QP, lower_bound = 0, upper_bound = 1))

            set_normalized_coefficient(λ_sum[i], λ[i][end], 1)
            set_normalized_coefficient(λ_y[i], λ[i][end], val)
        end

        delete.(QP, λ_SOS)
        unregister.(QP, :λ_SOS)


        iter +=1 

        println(repeat("=", 40))
        println("Iter: ", iter)
        println("Num Updates: ", length(split_vars))
        println("Objective: ", objval)
        println("Time: ", round(time() - t_start, digits = 2))

    end

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

    while true

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

        println("Iter: ", iter, "    ================================================")
        println("Level: ", subp_level)
        println("Upper Bound: ", global_ub)
        println("Lower Bound: ", global_lb)
        println("Remaining problems: ", length(open_model_bounds))
        println("================================================================")

    end

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

    # @constraint(SDP, sum(Qext .* Y) == (sum(y) + sum(c[i] * x[i] for i = 1:n))/2)

    @objective(SDP, Min, sum(Qext .* Y))


    subproblems = [(-Inf, [],[],[],[])] #(lb, F0, F1, Fy, Fz)

    global_ub = Inf
    global_lb = -Inf

    incumbent_sol = nothing

    iter = 0

    while true

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

        if round(global_lb, digits = 2) == round(global_ub, digits = 2)
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

        println("Iter: ", iter, "    ================================================")
        println("Upper Bound: ", global_ub)
        println("Lower Bound: ", global_lb)
        println("Remaining problems: ", length(subproblems))
        println("================================================================")

    end


end