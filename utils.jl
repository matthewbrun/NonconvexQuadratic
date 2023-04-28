using Random, LinearAlgebra

function build_box_concave_quadratic(n, d; seed = -1, type = :negative)
"""
Construct quadratic problem (Q, c) in dimension n, sparsity d
Inputs:
    n - number of variables in the problem
    d - sparsity of Q and c (sparsity of Q if Q indefinite, sparsity of Cholesky factor if pos/neg definite)
    seed - seed for random generation
    type - (:negative, :indefinite, :positive), determines definiteness of Qvec
Outputs:
    Q - symmetric matrix, size n x n of type with sparsity d and definiteness type
    c - vector, size n with sparsity d
"""

    #Set random seed
    if seed >= 0
        Random.seed!(seed)
    end

    ##Build random quadratic part

    #For indefinite matrices
    if type == :indefinite
        #Generate upper diagonal elements
        num_entries = Int(n*(n+1)/2)
        Qvec = rand(-50:50, num_entries)

        #Compute number of nonzeros from sparsity d
        nnz = Int(ceil(num_entries * d))
        nz = num_entries - nnz
        #Generate mesh to enforce sparsity
        mesh = shuffle([ones(nnz,1); zeros(nz,1)])

        #Apply mesh
        threshQvec = mesh .* Qvec

        #Enter elements into upper diagonal of matrix
        Q = zeros(n,n)
        itx = 0
        for i = 1:n
            for j = i:n
                itx += 1
                Q[i,j] = threshQvec[itx]
            end
        end

        #Convert to symmetric form
        Q = Symmetric(Q)
    end

    #For semidefinite matrices
    if type in [:negative, :positive]

        #Generate random cholesky factor
        B = rand(-7:7, (n,n))

        #Compute number of nonzeros in factor
        nnz = Int(ceil(n^2 * d))
        nz = n^2 - nnz
        #Generate mesh
        mesh = reshape(shuffle([ones(nnz,1); zeros(nz,1)]), (n,n))

        #Apply mesh to cholesky and convert to semidefinite matrix
        if type == :positive
            Q = (mesh.*B)' * (mesh.*B)
        elseif type == :negative
            Q = -(mesh.*B)' * (mesh.*B)
        end
        

    end

    ##Build random linear part
    cvec = rand(-50:50, n)

    #Compute number of nonzeros
    nnz = Int(ceil(n * d))
    nz = n - nnz
    #Build mesh
    mesh = shuffle([ones(nnz,1); zeros(nz,1)])

    #Apply mesh to vector 
    c = mesh .* cvec

    return Q, c

end