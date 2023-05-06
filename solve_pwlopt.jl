using JuMP, Gurobi, LinearAlgebra, MosekTools, InvertedIndices

include("utils.jl")
include("pwlopt.jl")

#TODO: warmstart from best incumbent solution
#TODO: warmstarting works poorly when revisiting previous solution

const LAMBDA_TOL = 1e-6

function adaptive_pwl(Q, c; branching = :SOS2, warmstart = true, breakpoint_management = :none, max_breakpoints = 4, large_decrease_threshold = 1/2, custom_termination = true)

    # breakpoint_management in [:aggressive, :standard, :none]
    # branching in [:SOS2, :CC, :MC, :IC, :DCC]

    if !(breakpoint_management in [:aggressive, :standard, :none])
        error("Unsupported breakpoint management strategy")
    end

    n = length(c)

    #Compute eigendecomposition
    eig = eigen(Q)
    B = eig.vectors
    Σ = eig.values

    split_points = [[sum((B[:,i] .<= 0) .* B[:,i]), sum((B[:,i] .>= 0) .* B[:,i])] for i = 1:n]

    iter = 0
    stop_iter = false

    solutions = []
    sol_objs = []

    upper_bound = Inf
    lower_bound = -Inf

    t_start = time()
    while !stop_iter

        QP = direct_model(optimizer_with_attributes(
            Gurobi.Optimizer, 
            MOI.Silent() => false,
            "MIPFocus" => 2
        ))
    
        @variable(QP, x[1:n] >= 0)
        @variable(QP, s_ub[1:n] >= 0)
    
        @variable(QP, y[1:n])
    
        @constraint(QP, x .+ s_ub .== 1)
    
        @constraint(QP, y .== B'*x)

        sort_order = [sortperm(split_points[i]) for i = 1:n]
        sorted_splits = [split_points[i][sort_order[i]] for i = 1:n]
        sorted_vals = [Σ[i] * sorted_splits[i].^2 for i = 1:n]

        if iter == 0
            z = [@variable(QP) for i = 1:n]
            slope = [Σ[i] * (sorted_splits[i][2]^2 - sorted_splits[i][1]^2)/(sorted_splits[i][2] - sorted_splits[i][1]) for i = 1:n]
            @constraint(QP, [i = 1:n], z[i] == slope[i] * (y[i] - sorted_splits[i][1]) + Σ[i] * sorted_splits[i][1]^2)
        else
            z = [piecewiselinear(QP, y[i], sorted_splits[i], sorted_vals[i], method = branching) for i = 1:n]
        end

        @objective(QP, Min, sum(z) + sum(c[i] * x[i] for i = 1:n))

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
        lower_bound = max(grb_lb, lower_bound)

        stop_iter = true
        split_vars = []
        for i = 1:n
            actual_objval = Σ[i] * value.(y[i]).^2
            compute_objval = value.(z[i])
            if abs((actual_objval - compute_objval)) > 1e-3
                stop_iter = false
                push!(split_vars, (i, value.(y[i])))
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

        println(repeat("=", 40))
        println("Iter: ", iter)
        println("Num Updates: ", length(split_vars))
        println("Objective: ", objval)
        println("Lower Bound: ", lower_bound)
        println("Upper Bound: ", upper_bound)
        println("Time: ", round(time() - t_start, digits = 2))

    end

end