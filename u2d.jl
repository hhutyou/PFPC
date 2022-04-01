function u2d(u::Array{T,1},Hn1::Array{T,2}) where T<:Float64
    epsilon = zeros(Float64,6,8*nel)
    ##
    UU = u[edofMat]
    for iel=1:nel
        epsilon[:,8*(iel-1)+1:8*iel] = reshape(view(Bu,:,24*(iel-1)+1:24*iel)*view(UU,iel,:),6,8)
    end
    Hn1 = Hn1_comp!(Hn1,epsilon)
    Bterm = kron(gc*ls,ones(8,nel))
    Nterm = kron(gc/ls,ones(8,nel)) .+ 2.0 * Hn1
    sKKd_B = SharedArray{Float64,2}(64,nel)
    sKKd_D = SharedArray{Float64,2}(64,nel)
    sFD = SharedArray{Float64,2}(8,nel)
    @sync @distributed for iel=1:nel
        sKKd_B[:,iel] = view(Bdp,:,8*(iel-1)+1:8*iel) .*  kron(detjacob[:,iel]',ones(64,1)) * view(Bterm,:,iel)
        sKKd_D[:,iel] = view(Ndp,:,8*(iel-1)+1:8*iel) .* kron(detjacob[:,iel]',ones(64,1)) * view(Nterm,:,iel)
        sFD[:,iel] = 2.0 * view(Nd,:,8*(iel-1)+1:8*iel)' .* kron(detjacob[:,iel]',ones(8,1)) * view(Hn1,:,iel)
    end
    # sKd_B::Array{Float64,1} = reshape(sKKd_B,16*nel)
    # sKd_D::Array{Float64,1} = reshape(sKKd_D,16*nel)
    # sKd = sKd_B+sKd_D
    # sFd = reshape(sFD,4*nel)
    Kd::SparseMatrixCSC = sparse(iKd, jKd, vec(sKKd_B).+vec(sKKd_D))#::SparseMatrixCSC
    # Kd::SparseMatrixCSC = sparse(iKd,jKd,reshape(sKKd_D,16*nel))
    Fd::SparseVector = sparse(iFd, jFd, vec(sFD))#::SparseVector
    # d1 = Kd \ Array(Fd)
    psu = MKLPardisoSolver()
    set_nprocs!(psu, ncore)
    ##
    return solve(psu, Kd, Array(Fd)), Hn1
end
