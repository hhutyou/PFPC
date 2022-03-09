function plasticity!(σc::Array{T},dg1::Array{T},idxCom) where T<:Float64
    Ip=[1/3,1/3,0.0]
    itol=1e-6
    imaxit=100
    # m0(x) = (1.0.-x).^2 ./ (ka .- (ka .- 1.0).*(1.0 .- x).^2)
    gd(x) = (1.0.-x).^2 ./ (ka .- ka.*(1.0 .- x).^2)
    # Eᵖ_old = deepcopy(Eᵖ)
    # nel::Int=size(element,1)
    F = Array{Float64}(undef,4*nel)
    δλ = zeros(Float64,4*nel)
    δεᵖ = zeros(Float64,3,4*nel)
    # Eᵖ = Array{Float64}(undef,4*nel)
    # σ = Array{Float64}(undef,3,4*nel)
    # Eᵖ .= Eᵖ
    P, J2, s = invariant(σc,v,planetype)
    F .= sqrt.(2.0*J2).+ AA.*P
    # F[Mat_ind1] = sqrt.(2.0*J2[Mat_ind1]).+ A01.*P[Mat_ind1]
    # F[Mat_ind2] = sqrt.(2.0*J2[Mat_ind2]).+ A0.*P[Mat_ind2]
    idxYield = findall(F.>itol)
    # UidxCY = intersect(idxYield)
    UidxCY = intersect(idxYield,idxCom)
    # @info "isempty(UidxCY)=$(isempty(UidxCY))"
    # UidxCY2 = intersect(idxYield,Mat_ind2,idxCom)
    # if ! isempty(UidxCY)
    #     @info "----材料1屈服调整"
    # end
    # if ! isempty(UidxCY2)
    #     @info "----材料2屈服调整"
    # end
    if isempty(UidxCY)
        @info "----未屈服"
        return δεᵖ
    else
        ##并行计算数组初始化
        # @info "----塑性调整"
        if !isempty(UidxCY)
            δλC::SharedArray = δλ[UidxCY]
            FC::SharedArray = F[UidxCY]
            J2C::SharedArray = J2[UidxCY]
            PC::SharedArray = P[UidxCY]
            sC::SharedArray = s[:,UidxCY]
            σC::SharedArray = σc[:,UidxCY]
            # σᵗC::SharedArray =  copy(σC)
            δεᵖC::SharedArray = δεᵖ[:,UidxCY]
            dg1C = dg1[UidxCY]
            GC = G[UidxCY]
            KvC = Kv[UidxCY]
            AC = AA[UidxCY]
            @sync @distributed for num = 1:size(UidxCY,1)#@distributed (+)
                δλC[num] = 0.0
                ierr=1.0
                init=0
                # JJ2 = sigma_J2(σC[:,num],v,planetype)
                while ierr>itol && init<imaxit
                    init += 1
                    # @info ("---塑性迭代init=$init")
                    d = -2.0*(1.0+gd(dg1C[num]))*GC[num]-(1.0+gd(dg1C[num]))*KvC[num]*(AC[num])^2
                    ##残余屈服值对塑性因子的导数%%参考computation methods for plasticity-P332
                    δλC[num]=δλC[num]-FC[num]/d
                    ##检查收敛性
                    FC[num]=sqrt(2.0*J2C[num])-2.0*(1.0+gd(dg1C[num]))*GC[num]*δλC[num]+AC[num]*(PC[num]-(1.0+gd(dg1C[num]))*KvC[num]*AC[num]*δλC[num])
                    ierr = abs(FC[num])
                end
                # σC[:,num] = σᵗC[:,num]-δλC[num].*(1.0-gd(dg1C[num]))*(2.0*GC[num]*sC[:,num]/sqrt(2.0*J2C[num])+AC[num]*KvC[num]*Ip)
                # sigmaC[:,num] = sigmaᵗC[:,num]-δλC[num].*(2.0*GC[num]*sC[:,num]/sqrt(2*J2C[num])+AC[num]*KvC[num]*Ip)
                δεᵖC[:,num] = δλC[num]*(sC[:,num]/sqrt(2.0*J2C[num])+AC[num]*Ip)
            end
            # σc[:,UidxCY] .=   σC
            # sigma[:,UidxCY] .= sdata(sigmaC)
            δεᵖ[1:2,UidxCY] = sdata(δεᵖC)[1:2,:]
            δεᵖ[3,UidxCY] = 2.0*sdata(δεᵖC)[3,:]
        end
    end
    return δεᵖ
end
