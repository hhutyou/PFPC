function d2u(u_increment::T2,stp::T1,u::Array{T2},d1::Array{T2}) where {T1<:Int64, T2<:Float64}
    m0(x) = (1.0 .- k).*(1.0.-x).^2 .+ k
    # epsilon = zeros(Float64,3,4*nel)
    epsilon = Array{Float64,2}(undef,3,4*nel)
    dg1=Array{Float64,1}(undef,4*nel)
    ##
    UU = u[edofMat]
    for iel=1:nel
        epsilon[:,4*(iel-1)+1:4*iel] = reshape(view(Bu, :, 8*(iel-1)+1:8*iel) * view(UU, iel, :),3,4)
        dg1[4*(iel-1)+1:4*iel] = view(Nd, :, 4*(iel-1)+1:4*iel) * view(d1, dedofMat[iel,:])
    end
    # DK = kron(m0.(dg1), ones(1,9))' .* (3Kv0 .* kron(hc.(operator_tr(epsilon,planetype)), ones(1,9))' .* Jb .+ 2μ0 .* Kb  ) .+ 3Kv0 .* kron(mhc.(operator_tr(epsilon,planetype)), ones(1,9))' .* Jb
    DK = 3Kv0 .* Jb .+ 2μ0 .* Kb .+ kron((m0.(dg1) .- 1.0), ones(1,9))' .* (3Kv0 .* kron(hc.(operator_tr(epsilon,planetype)), ones(1,9))' .* Jb .+ 2μ0 .* Kb  )
    KK = Kmatrix(element, Bu, detjacob,DK,iK,jK)
    u[loaddofs] .= u_increment
    u[freedofs] .= -KK[freedofs, freedofs] \ (KK[freedofs, loaddofs] * (u[loaddofs]))
    ##
    return u, KK
end
