function u2d(u::Array{T,1},Hn1::Array{T,2}) where T<:Float64
    epsilon = zeros(Float64,3,9*nel)
    ##
    UU = u[edofMat]
    for iel=1:nel
        epsilon[:,9*(iel-1)+1:9*iel] = reshape(view(Bu,:,16*(iel-1)+1:16*iel)*view(UU,iel,:),3,9)
    end
    Hn1 = Hn1_comp!(Hn1,epsilon)
    Bterm = kron(gc*ls,ones(9,nel))
    Nterm = kron(gc/ls,ones(9,nel))+2.0*Hn1
    sKKd_B = SharedArray{Float64,2}(16,nel)
    sKKd_D = SharedArray{Float64,2}(16,nel)
    sFD = SharedArray{Float64,2}(4,nel)
    @sync @distributed for iel=1:nel
        sKKd_B[:,iel] = view(Bdp,:,9*(iel-1)+1:9*iel) .*  kron(detjacob_d[:,iel]',ones(16,1)) * view(Bterm,:,iel)
        sKKd_D[:,iel] = view(Ndp,:,9*(iel-1)+1:9*iel) .* kron(detjacob_d[:,iel]',ones(16,1)) * view(Nterm,:,iel)
        sFD[:,iel] = 2.0 * view(Nd,:,4*(iel-1)+1:4*iel)' .* kron(detjacob_d[:,iel]',ones(4,1)) * view(Hn1,:,iel)
    end
    # sKd_B::Array{Float64,1} = reshape(sKKd_B,16*nel)
    # sKd_D::Array{Float64,1} = reshape(sKKd_D,16*nel)
    # sKd = sKd_B+sKd_D
    # sFd = reshape(sFD,4*nel)
    Kd::SparseMatrixCSC = sparse(iKd,jKd,reshape(sKKd_B,16*nel)+reshape(sKKd_D,16*nel))#::SparseMatrixCSC
    # Kd::SparseMatrixCSC = sparse(iKd,jKd,reshape(sKKd_D,16*nel))
    Fd::SparseVector = sparse(iFd,jFd,reshape(sFD,4*nel))#::SparseVector
    d1 = Kd \ Array(Fd)
    ##
    return vec(d1), Hn1
end
