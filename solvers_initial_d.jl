function initial_d(freedofs_d::Array{T}, loaddofs_d::Array{T}, lss::Float64, d0::Float64) where T<:Int64
    ## Initialization_d
    @info "Initialization_d"
    δd = zeros(Float64,nnode)
    rd = zeros(Float64,nnode)
    dd = zeros(Float64,nnode)
    d_old = zeros(Float64,nnode)
    dg1=zeros(Float64,4*nel)
    dd[loaddofs_d] .= 1.0
    ##
    Bterm = kron(lss^2,ones(4,nel))
    Nterm = kron(1.0,ones(4,nel))
    sKKd_B=SharedArray{Float64,2}(16,nel)
    sKKd_D=SharedArray{Float64,2}(16,nel)
    sFD=SharedArray{Float64,2}(4,nel)
    @sync @distributed for iel=1:nel
        sKKd_B[:,iel]=Bdp[:,4*(iel-1)+1:4*iel].*kron(detjacob[:,iel]',ones(16,1))*Bterm[:,iel]
        sKKd_D[:,iel]=Ndp[:,4*(iel-1)+1:4*iel].*kron(detjacob[:,iel]',ones(16,1))*Nterm[:,iel]
    end
    Kd::SparseMatrixCSC = sparse(iKd,jKd,reshape(sKKd_B,16*nel)+reshape(sKKd_D,16*nel))#::SparseMatrixCSC
    #
    rd[freedofs_d] .= Kd[freedofs_d,loaddofs_d]*dd[loaddofs_d]
    δd[freedofs_d] = -Kd[freedofs_d,freedofs_d]\rd[freedofs_d]
    dd .= d_old .+ δd .+ dd .+ d0
    #
    for iel = 1:nel
        dg1[4*(iel-1)+1:4*iel]=Nd[:,4*(iel-1)+1:4*iel]*dd[dedofMat[iel,:]]
    end
    return dd, dg1
end
