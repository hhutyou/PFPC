function u_calcu(uu::Array{T2},du::Array{T2},d1::Array{T2},nit::T1,stp::T1) where {T1<:Int64, T2<:Float64}
    epsilon = SharedArray{Float64,2}(3,9*nel)  #gauss点应变
    dg1=SharedArray{Float64,1}(9*nel)
    UU = uu[edofMat]
    DD = d1[dedofMat]
    @sync @distributed for iel=1:nel
        epsilon[:,9*(iel-1)+1:9*iel] = reshape(view(Bu, :, 16*(iel-1)+1:16*iel) * view(UU, iel, :),3,9)
        dg1[9*(iel-1)+1:9*iel] = view(Nd, :, 4*(iel-1)+1:4*iel) * view(DD, iel,:)
    end
    #计算弹性矩阵C_set91   C 2*2*2*(2*4nel) → C 3*(3*4nel) → C 9*(1*4nel)
    C_set91 = Array{Float64,2}(undef,9,9nel); C_set91 = repeat(Cbulk_vec,1,9nel)
    C_set91  = C_update(C_set91,d1,Array(dg1), Array(epsilon) ,element_central)
    uuK = SharedArray{Float64,2}(256,size(element,1))
    uuK = Juu_matrix(uuK,C_set91)
    Juu::SparseMatrixCSC = sparse(iKu,jKu,uuK[:])#::SparseMatrixCSC
    #计算δu
    err=1; nit_u=0
    δu = zeros(Float64,2*nnode)
    rr = zeros(Float64,2*nnode)
    u_ref = deepcopy(δu)
    if nit == 1
        rr[freedofs_u] .= Juu[freedofs_u,loaddofs_u]*du[loaddofs_u]
        uu .= uu .+ du
    else
        rr .= force_int(node,element,uuK,uu,edofMat)
    end
    r_ref = deepcopy(rr)
    psu = MKLPardisoSolver()
    while   nit_u<maxit_u
        nit_u+=1
        @info "-u迭代步=: $nit_u"
        δu[freedofs_u] .= - solve(psu,Juu[freedofs_u,freedofs_u],rr[freedofs_u] )
        #δu[freedofs_u] =  - Juu[freedofs_u,freedofs_u] \ rr[freedofs_u]
        if nit_u == 1
            u_ref .= δu
        end
        uu .= uu .+ δu
        # if nit==1 && nit_u==1
        #     #r[freedofs_u] = Juu[freedofs_u,loaddofs_u]*du[loaddofs_u]
        #
        #     rr[freedofs_u] = Juu[freedofs_u,loaddofs_u]*du[loaddofs_u] # + f_int_prev[freedofs_u]
        #     δu[freedofs_u] .= - Juu[freedofs_u,freedofs_u] \ rr[freedofs_u]
        #     # δu[freedofs_u] .= - solve(psu,Juu[freedofs_u,freedofs_u],rr[freedofs_u] )
        #     r_ref .= rr
        #     #第一次迭代加上位移加载值
        #     # u_old[loaddofs_u] .= u_old[loaddofs_u] + du[loaddofs_u]
        #     uu .= u_old .+ du .+ δu
        # elseif nit_u>1
        #     δu[freedofs_u] =  - Juu[freedofs_u,freedofs_u]\rr[freedofs_u] #原式
        #     # δu[freedofs_u] .= - solve(psu, Juu[freedofs_u,freedofs_u],rr[freedofs_u] )
        #     uu .= u_old .+ δu
        # else
        #     nothing
        # end
        # epsilon = Array(epsilon)
        # f_int = force_int(node,element,uuK,u,edofMat)
        rr .= force_int(node,element,uuK,uu,edofMat)
        err = abs(rr' * δu) / abs(r_ref' * u_ref)
        #err = norm(δu[freedofs_u],Inf)/norm(uu[freedofs_u],Inf)
        @info "err_u = $err"
        err < tol_u ? break : nothing
        #计算弹性矩阵C_set91   C 2*2*2*(2*4nel) → C 3*(3*4nel) → C 9*(1*4nel)
        UU .= uu[edofMat]
        @sync @distributed for iel=1:nel
            epsilon[:,9*(iel-1)+1:9*iel] = reshape(view(Bu, :, 16*(iel-1)+1:16*iel) * view(UU, iel, :),3,9)
        end
        C_set91 = Array{Float64,2}(undef,9,9nel); C_set91 = repeat(Cbulk_vec,1,9nel)
        C_set91  = C_update(C_set91,d1,Array(dg1), Array(epsilon) ,element_central)
        uuK = Juu_matrix(uuK,C_set91)
        Juu = sparse(iKu,jKu,uuK[:])#::SparseMatrixCSC
    end

    #单元应力2023.01 裂纹单元的应力不一致？
    #输出gauss应力
    UU = uu[edofMat]
    @sync @distributed for iel=1:nel
        epsilon[:,9*(iel-1)+1:9*iel] = reshape(view(Bu, :, 16*(iel-1)+1:16*iel) * view(UU, iel, :),3,9)
    end
    σ = SharedArray{Float64,2}(3,9*nel) #gauss
    @sync @distributed for iel=1:9nel
        σ[:,iel] = reshape(C_set91[:,iel],3,3) * epsilon[:,iel]
    end
    σx = sum(reshape(σ[1,:],9,nel),dims=1)
    σy = sum(reshape(σ[2,:],9,nel),dims=1)
    τxy = sum(reshape(σ[3,:],9,nel),dims=1)
    σEle = [σx; σy; τxy]' #nel行3列
    println(size(σEle))
    fid = open("σEle.dat", "w")
    writedlm(fid, σEle[EleCrack,1:3])
    close(fid)
    σ45 = 0.5*(σEle[EleCrack,1]+σEle[EleCrack,2])-σEle[EleCrack,3]
    τ45 = 0.5*(σEle[EleCrack,1]-σEle[EleCrack,2])
    fid = open("σ45.dat", "w")
    writedlm(fid, σ45)
    close(fid)
    fid = open("τ45.dat", "w")
    writedlm(fid, τ45)
    close(fid)
    return uu
end

#=
    用于水力劈裂的
    #计算系数
    αc_gauss, φc_gauss, Mc1_gauss = poroelasticity_parm_gauss(dg1,Mc1_gauss,αc_gauss)
    kε_mat,dkdε1 = permiability(epsilon)
    dε1dεxx,dε1dεyy,dε1dεxy = dε1dε(epsilon)
    #组装矩阵
    uuK = Juu_matrix(element, Bu, detjacob,DK)
    ppK,ppK_2 = Jpp_matrix(element, Bp, Npp, detjacob,kε_mat,Mc1_gauss,Mc1_gauss_prevstp)
    upK = Jup_matrix(element, Bu, Np, detjacob,αc_gauss)
    puK = Jpu_matrix(element, Bu_xx,Bu_yy,Bu_xy,dε1dεxx,dε1dεyy,dε1dεxy,Np,Bp,dkdε1,Bu_vol,detjacob,αc_gauss,αc_gauss_prevstp,var_up_old,pdofMat)
    Jglobal = Jele_matrix(node,element,uuK,upK,puK,ppK)
    R_up = Rup_matrix(node,element,uuK,ppK_2,upK,var_up_old,var_up_prevstp,Np,Bu_vol,detjacob,Mc1_gauss,Mc1_gauss_prevstp,αc_gauss,αc_gauss_prevstp,udofMat,pdofMat,f_ext,fp_ext)
    δup = zeros(Float64,3*nnode)
    #Jglobal[freedofs_up,freedofs_up]*δup[freedofs_up] + Jglobal[freedofs_up,loaddofs_p]*δup[loaddofs_p] + R_up[freedofs_up] = 0
    #其中δup[loaddofs_p]=0
    δup[freedofs_up] = Jglobal[freedofs_up,freedofs_up]\(-R_up[freedofs_up])
    #δup = Jglobal \ (-R_up)
    var_up = var_up_old .+ δup
    return var_up,Mc1_gauss,αc_gauss
=#
