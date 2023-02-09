#应力判断是高斯点值
#update σ & ℂ
function C_update(C_set91::Array{T2,2},d1::Array{T2,1},dg1::Array{T2,1},epsilon::Array{T2,2},element_central::Array{T2,2}) where {T1<:Int64,T2<:Float64}
    #接触情况分类bulk,nocontact,stick,slip
    σ = SharedArray{Float64,2}(3, 9nel)
    @sync @distributed for i = 1:9*nel
        σ[:, i] = Cbulk33 * epsilon[:, i] #应力预赋值为σbulk, σ = [σxx;σyy;σxy=τxy]
    end
    #cond_bulk = findall(dg1 .<= inter_detec)#bulk region index 1，无需更改ℂ
    cond_inter = findall(dg1 .> inter_detec)#interface region
    #nocontact-----------------------------------------------------------------------------------------------------------------
    epsilonN = Array{Float64,1}(undef, size(cond_inter, 1))
    #n_vec, m_vec = crack_regularize2d(d1, cond_inter, element_central)
    n_vec, m_vec = crack_orientation(d1, cond_inter, element_central)

    nn_vec = cal_nn(cond_inter, n_vec)
    epsilonN = sum(epsilon[:, cond_inter] .* nn_vec[:, cond_inter], dims = 1)
    cond_nocontact = cond_inter[findall(epsilonN' .>= 0)] #no-contact需更改ℂ或者σ ,找出dg1中的索引号
    #@show cond_nocontact
    σ[:, cond_nocontact] = ((1.0 .- kk) * (1.0 .- dg1[cond_nocontact]).^2 .+ kk)' .* σ[:, cond_nocontact] #σ=g(d)*σbulk
    C_set91[:,cond_nocontact] = ((1.0 .- kk)*(1.0.-dg1[cond_nocontact]).^2 .+ kk)' .* Cbulk_vec  #（1-d)^2要横排
    #contact--------------------------------------------------------------------------------------------------------------------
    #stick&slip-------------------------------------------------------------------------------------------------------------------
    cond_contact = setdiff(cond_inter, cond_nocontact)
    τbulk = Array{Float64,1}(undef, 9 * nel)
    pNbulk = Array{Float64,1}(undef, 9 * nel)
    f = Array{Float64,1}(undef, size(cond_contact, 1))

    nm_vec = cal_nm(cond_contact, n_vec, m_vec)
    τbulk[cond_contact] = sum(σ[:, cond_contact] .* nm_vec[:, cond_contact], dims = 1)
    pNbulk[cond_contact] = -sum(σ[:, cond_contact] .* (nn_vec[:, cond_contact] .* [1; 1; 2]), dims = 1)  #nn_vec第三行缩小了两倍
#=     pNposi = size(findall(pNbulk[cond_contact].>0));pNnega = size(findall(pNbulk[cond_contact].<=0))
    τbulkposi = size(findall(τbulk[cond_contact].>0));τbulknega = size(findall(τbulk[cond_contact].<=0))
    @show pNposi,pNnega
    @show τbulkposi,τbulknega =#

    pNEle = Array{Float64,1}(undef, nel)
    pNbulk9 = reshape(pNbulk,9,nel)
    pNEle = ((sum(pNbulk9,dims=1))/9)'

    fid = open("pNEle.dat", "w")
    writedlm(fid, pNEle[EleCrack])
    close(fid)

    f = abs.(τbulk[cond_contact]) .- μ_fric .* (pNbulk[cond_contact])
    #pNbulk应>0
    cond_stick = cond_contact[findall(f .< 0)] #2.2.1  contact stick 无需更改ℂ或者σ
    cond_slip = setdiff(cond_contact, cond_stick) #2.2.2 contact slip   需更改ℂ或者σ
    @info "cond_nocontact: $(size(cond_nocontact)),cond_contact: $(size(cond_contact)), cond_stick: $(size(cond_stick)), cond_slip: $(size(cond_slip))"

    #输出slip所属于的单元
    Ele_slip = unique(div.(cond_slip .-1 ,9) .+ 1)

    fid = open("Ele_slip.dat", "w")
    writedlm(fid, Ele_slip)
    close(fid)

    nmmn_mat = Array{Float64,2}(undef,2,2*9nel) #nmmn向量转换格式，用于四阶张量运算
    for i in cond_slip
        nmmn_mat[1:2,2i-1:2i] = [2*nm_vec[1,i] nm_vec[3,i]; nm_vec[3,i] 2*nm_vec[2,i]]
    end
    nn_mat = Array{Float64,2}(undef,2,2*9nel) #nn向量转换格式，用于四阶张量运算
    for i in cond_slip
        nn_mat[1:2,2i-1:2i] = [nn_vec[1,i] nn_vec[3,i];nn_vec[3,i] nn_vec[2,i]]
    end
    Cf = Array{Float64,4}(undef,2,2,2,2)#loop内临时存储
    Cτ = Array{Float64,4}(undef,2,2,2,2)#loop内临时存储
    Ctemp = Array{Float64,4}(undef,2,2,2,2);Ctemp91 = Array{Float64,2}(undef,9,1)#loop内临时存储
    for m in cond_slip  #m就是高斯点序号了
        for i=1:2,j=1:2,k=1:2,l=1:2
            Cf[i,j,k,l] = -sign(τbulk[m])*μ_fric*( λ0*nmmn_mat[:,2m-1:2m][i,j]*δ[k,l] + 2*G0*nmmn_mat[:,2m-1:2m][i,j]*nn_mat[1:2,2m-1:2m][k,l] )

            Cτ[i,j,k,l] = G0*(nmmn_mat[:,2m-1:2m][i,j]*nmmn_mat[:,2m-1:2m][k,l])
        end # for
        Ctemp = Cf-Cτ
        #降阶
        Ctemp91 = [Ctemp[1,1,1,1];Ctemp[2,2,1,1];Ctemp[1,2,1,1];Ctemp[1,1,2,2];Ctemp[2,2,2,2];Ctemp[1,2,2,2];Ctemp[1,1,1,2];Ctemp[2,2,1,2];Ctemp[1,2,1,2]]
        C_set91[1:9,m] .+= (1.0-((1.0 .- kk)*(1.0-dg1[m])^2 +kk)) .* Ctemp91 
    end
    return C_set91  #gauss
end # function



#=     ϵ = Array{Float64,2}(undef,2,2)
    ϵ = [1 2;3 4]
    σ = [sum(Cbulk[1,1,:,:].*ϵ) sum(Cbulk[1,2,:,:].*ϵ);
         sum(Cbulk[2,1,:,:].*ϵ) sum(Cbulk[2,2,:,:].*ϵ)] =#
