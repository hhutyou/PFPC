function Hn1_3D!(Hn1::Array{T,2},epsilon_gauss::Array{T,2}) where T<:Float64
    ψᵉ = zeros(Float64,4*nel)
        #strain energy decomposition
    epsilon_tr = operator_tr(epsilon_gauss)
    epsilon_dev = operator_dev(epsilon_gauss,epsilon_tr)

    ψᵉ = 0.5.*Kv0.*(heaviside.(epsilon_tr)).^2 
    .+ μ0.*((epsilon_dev[1,:]).^2 .+ (epsilon_dev[2,:]).^2 .+ (epsilon_dev[3,:]).^2 
    .+ 0.5.*(epsilon_dev[4,:]).^2 .+ 0.5.*(epsilon_dev[5,:]).^2 .+ 0.5.*(epsilon_dev[6,:]).^2 )  #Amor
    
    H_initial =zeros(Float64,4*nel) 
#=     ele_1 = [1:1:62 ...]  #损伤单元号
    ele_2 = union(ele_1*9, ele_1*9 .-1, ele_1*9 .-2, ele_1*9 .-3)  #gauss点
    H_initial[ele_2] .= 1e18   =#

    D1::Array{T,1} = max.(H_initial,ψᵉ)
    Hn1::Array{T,2} = max.(reshape(D1,4,nel),Hn1) 


    return Hn1
end

function Hn1_comp!(Hn1::Array{T,2},epsilon_gauss::Array{T,2}) where T<:Float64
    ψᵉ = zeros(Float64,9*nel)
    #strain energy decomposition
    epsilon_tr = operator_tr(epsilon_gauss,"plane-strain")
    epsilon_dev = operator_dev(epsilon_gauss,epsilon_tr,"plane-strain")

    #ψᵉ = μ0.*((epsilon_dev[1,:]).^2 .+ (epsilon_dev[2,:]).^2 .+ 0.5.*(epsilon_dev[3,:]).^2)  #Choo
    ψᵉ = 0.5.*Kv0.*(heaviside.(epsilon_tr)).^2 .+ μ0.*((epsilon_dev[1,:]).^2 .+ (epsilon_dev[2,:]).^2 .+ 0.5.*(epsilon_dev[3,:]).^2)  #Amor

#= #hybrid formulation--Rankine
    σ = SharedArray{Float64,2}(3, 4nel)
    @sync @distributed for i = 1:4*nel
        σ[:, i] = Cbulk33 * epsilon_gauss[:, i]
    end
    σ = Array(σ)
    σ1 = principle1(σ,"plane-strain") #求最大主应力
    ψᵉ = (heaviside.(σ1)).^2 ./(2E0) =#
#hybrid formulation--modified Mises
    #  η = 10.0 #10-20 岩石压缩拉伸强度比
    # ϵeq = epsilon_eq(epsilon_gauss,η)
    # ψᵉ .= E0.*(ϵeq).^2 /2
    #应变历史函数
    H_plus = zeros(Float64,9*nel)
    H_plus .= ψᵉ  
    H_initial =zeros(Float64,9*nel)
     
#=     ele_1 = [1:1:62 ...]  #损伤单元号 Q8和Q4一样
    ele_2 = union(ele_1*9, ele_1*9 .-1, ele_1*9 .-2, ele_1*9 .-3, ele_1*9 .-4, ele_1*9 .-5, ele_1*9 .-6, ele_1*9 .-7, ele_1*9 .-8)  #gauss点
    H_initial[ele_2] .= 1e18   =#

    D1::Array{T,1} = max.(H_initial,H_plus)
    Hn1::Array{T,2} = max.(reshape(D1,9,nel),Hn1)  #与历史值比较
    return Hn1
end
#= 水力压裂
function Hn1_comp!(Hn1::Array{T,2},epsilon::Array{T,2},p_gauss::Array{T,2}) where T<:Float64
    ψᵉ = zeros(Float64,4*nel)
    # ψᵉ .= Kv0/2.0 .* (heaviside.(operator_tr(εᵉ,planetype))).^2 .- μ0/3.0 .* (operator_tr(εᵉ,planetype)).^2 .+
    #     μ0*((εᵉ[1,:]).^2 .+ (εᵉ[2,:]).^2 .+ 0.5.*(εᵉ[3,:]).^2)
#   法1
#    ψᵉ .= Kv0/2.0 .* (operator_tr(epsilon,planetype)).^2 .- μ0/3.0 .* (operator_tr(epsilon,planetype)).^2 .+
#        μ0*((epsilon[1,:]).^2 .+ (epsilon[2,:]).^2 .+ 0.5.*(epsilon[3,:]).^2)
    #法2
    eps_plus = SharedArray{Float64,2}(3,4*nel)#逐个点求epsilon+
    @sync @distributed for i = 1:4*nel
        D,V = eigen([epsilon[1,i] epsilon[3,i]/2; epsilon[3,i]/2 epsilon[2,i]]);
        A = (D[1]+abs(D[1]))/2*V[:,1]*V[:,1]' + (D[2]+abs(D[2]))/2*V[:,2]*V[:,2]';
        eps_plus[:,i] = [A[1,1]; A[2,2]; 2*A[1,2]];
    end
    eps_plus = sdata(eps_plus)
    ψᵉ .= Kv0/2.0 .* (operator_tr(epsilon,planetype)).^2 .- μ0/3.0 .* (operator_tr(epsilon,planetype)).^2 .+
        μ0*((eps_plus[1,:]).^2 .+ (eps_plus[2,:]).^2 .+ 0.5.*(eps_plus[3,:]).^2)
    H_plus = zeros(Float64,4*nel)
    H_plus = 2*ψᵉ .- p_gauss.^2 .*(  (1/Kf-1/Ks)*(1-φ0) + (Kv0/(Ks^2))   )
#    H_plus = ψᵉ
    D1::Array{T,1} = max.(0.0,H_plus)  #正数就是本身，负数就取0
    Hn1::Array{T,2} = max.(reshape(D1,4,nel),Hn1)  #与历史值比较
    return Hn1
end
=#

#=
function Hn1_comp!(H0::Array{T},Hn1::Array{T,2},ψᵖ::Array{T},CC::Array{T}) where T<:Float64
    D1::Array{T,1} = max.(0.0, H0 .+ ψᵖ./(hc.(CC).*gc1 .+ (1.0.-hc.(CC)).*gc))
    # D1 = zeros(size(ψᵉ))
    Hn1::Array{T,2} = max.(reshape(D1,4,nel),Hn1)
    return Hn1
end
#
function Hn2_comp!(ψᵖ::Array{T,1},Hn2::Array{T,2}) where T<:Float64
    D1::Array{T,1} = max.(0,ψᵖ)
    Hn2::Array{T,2} = max.(reshape(D1,4,size(element,1)),Hn2)
    return  Hn2
end
=#
