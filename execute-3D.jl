
using Distributed,JLD2
addprocs(8-nprocs())
println("Running ",nprocs()," processes")
@everywhere using LinearAlgebra, SparseArrays, SharedArrays, DelimitedFiles, Pardiso
include("Mesh3D.jl")
@everywhere include("FemBase3D.jl")
include("boundary3D.jl") 
include("ShapeFuncTet.jl") 
include("Solvers3D-1.jl")
include("u2d3D.jl")
include("K_f_matrix3D.jl")
include("Hn.jl")
include("d2u3D.jl")
# loading&iterations
const u_inc1, u_inc2, step_total = 2e-4, 5e-5, 87 #200 Int(100+0.005/5e-5)
const maxit = 200
const maxit_u = 15
const tol = 1.0e-3
const tol_u = 0.0015

# phase field parameters
const ls,mesh_size, kk = 0.007,0.003, 1e-16
const gc = 2.7e-3 #2.7N/mm=2.7e-3kN/mm

# constitutive parameters    单位mm,GPa,kN
const inter_detec = 0.2
const μ_fric = 0.2 #friction coefficient
const threshold = 0.999 #损伤阈值
#const E0, v = 210.0 , 0.3
const λ0, μ0 = 121.15, 80.77 #λ0=E0*v/((1+v)*(1-2*v)),μ0=E0/(2*(1+v)) λ0=Kv0-2/3*G0
const G0, Kv0 = 80.77 , 175.0  #GPa  #G0=μ0,Kv0=E0/(3*(1-2*v))

#denote σ = [σx σy σz τyz τxz τxy]'; ε = [εx εy εz γyz γxz γxy]'
const Cbulk33 = [ Kv0+(4/3)G0 Kv0-(2/3)G0 Kv0-(2/3)G0 0 0 0
                  Kv0-(2/3)G0 Kv0+(4/3)G0 Kv0-(2/3)G0 0 0 0
                  Kv0-(2/3)G0 Kv0-(2/3)G0 Kv0+(4/3)G0 0 0 0
                  0             0            0        G0 0 0
                  0             0            0        0 G0 0
                  0             0            0        0 0 G0]
const Cbulk_vec = [Kv0+(4/3)G0; Kv0-(2/3)G0; Kv0-(2/3)G0; 0; 0; 0; Kv0-(2/3)G0; Kv0+(4/3)G0; Kv0-(2/3)G0; 0; 0; 0; Kv0-(2/3)G0; Kv0-(2/3)G0; Kv0+(4/3)G0; 0; 0; 0; 0; 0; 0; G0; 0; 0; 0; 0; 0; 0; G0; 0; 0; 0; 0; 0; 0; G0]

#const δ = [1 0 0;0 1 0;0 0 1]
#const Ih = [1; 1; 0;1; 1; 0;0; 0; 0];const Id = [2/3; -1/3; 0;-1/3; 2/3; 0;0; 0; 1/2] 
const tk6 = [  0.666667  -0.333333  -0.333333  0.0  0.0  0.0
              -0.333333   0.666667  -0.333333  0.0  0.0  0.0
              -0.333333  -0.333333   0.666667  0.0  0.0  0.0
               0.0        0.0        0.0       0.5  0.0  0.0
               0.0        0.0        0.0       0.0  0.5  0.0
               0.0        0.0        0.0       0.0  0.0  0.5]
const ti6 = [ 1.0  0.0  0.0  0.0  0.0  0.0
              0.0  1.0  0.0  0.0  0.0  0.0
              0.0  0.0  1.0  0.0  0.0  0.0
              0.0  0.0  0.0  1.0  0.0  0.0
              0.0  0.0  0.0  0.0  1.0  0.0
              0.0  0.0  0.0  0.0  0.0  1.0]         
const ti6_vec = ti6[:]                     
const tk6_vec = tk6[:]                     
 

const numd=step_total ##output number
const aa=Int.(step_total/numd)
begin ##初始化结果储存矩阵
    numD = Array{Float64,2}(undef,ncorner,numd); numD2 = Array{Float64,2}(undef,3*nnode,numd)#
    numD3 = Array{Float64,3}(undef,4,nel,numd); #numD4 = Array{Int32,2}(undef,maxit,step_total+1); numD5 = Array{Float64,2}(undef,maxit,step_total+1)
    Fload1 = Array{Float64}(undef,step_total+1); Uload1 = Array{Float64}(undef,step_total+1)
    Fload2 = Array{Float64}(undef,step_total+1); Uload2 = Array{Float64}(undef,step_total+1)
    iter_storage = Array{Float64}(undef,step_total+1); time_storge = Array{Float64}(undef,step_total+1)
end
##计算过程
    ## Confinement info:注意函数Fmat_ext
        conf = 0.0
    ## Sovle
    Fload1, Uload1, Fload2, Uload2, iter_storage, time_storge, numD, numD2, numD3 = solvers()
## Post-operation
    # include("out_E.jl")
    # nodeE = out_E(nel,node,element)
    # gNodeeps=pmap(out_accumulated_epsilonp, numD2[:,i] for i =1:numd)
    for opt=1:1:step_total
        A = [node[1:ncorner,:] numD[:, opt]]
        fid = open("d_data$(opt).dat", "w")
        StringVariable = "TITLE=\"2Dmodel\" VARIABLES=\"X\",\"Y\",\"Z\",\"d1\" ZONE N=$ncorner,E=$nel,F=FEPOINT,ET=TETRAHEDRON, "
        write(fid, StringVariable)
        writedlm(fid, A)
        writedlm(fid, element[:,1:4])
        close(fid)
    end
    # nx = (node[element[:,1],1] .+ node[element[:,2],1] .+ node[element[:,3],1] .+ node[element[:,4],1])/4.0
    # ny = (node[element[:,1],2] .+ node[element[:,2],2] .+ node[element[:,3],2] .+ node[element[:,4],2])/4.0
    # for opt=1:10:1
    #     A=[nx ny E[1:4:end,opt]]
    #     # A=[node out_accumulated_epsilonp(numD2[:,opt])]
    #     #numD[:,opt] out_accumulated_epsilonp(numD2[:,opt])
    #     # mat=[A;element]
    #     fid=open("Edata$opt.dat","w")
    #     StringVariable="TITLE=\"2Dmodel\" VARIABLES=\"X\",\"Y\",\"E\" ZONE I=$nel,J=$nel,DATAPACKING=BLOCK, "
    #     write(fid,StringVariable)
    #     # m,n=size(mat)
    #     writedlm(fid,A)
    #     # writedlm(fid,element)
    #     close(fid)
    # end
    # using Gadly
    fid2=open("data_U1_F1.dat","w")
    writedlm(fid2,[Uload1 Fload1])
    close(fid2)
    fid3=open("data_U2_F2.dat","w")
    writedlm(fid3,[Uload2 Fload2])
    close(fid3)
    # opt=plot(-Uload1,-Fload1,label="u1/u1=1"
    #     ,linewidth=1.0,color="red",xlabel="Displacement [mm]",ylabel="Force [MPa]"
    #     ,size=(500,320),legend=:topleft,dpi=300)
    # savefig("F1vsU1")
    ##data storage
    @save pwd()*"\\numD.jld2" numD
    @save pwd()*"\\numD2.jld2" numD2
    @save pwd()*"\\numD3.jld2" numD3
    @save pwd()*"\\iter_storage.jld2" iter_storage
    @save pwd()*"\\time_storge.jld2" time_storge
    fid=open("time_storge.dat","w")
    writedlm(fid,time_storge)
    close(fid)


#__________________________________________________________________________________________
#= 
##结果储存矩阵
numD = Array{Float64,2}(undef, ncorner, step_total)#d
numD2 = Array{Float64,2}(undef, 3*ncorner, step_total)#u


##开始计算
Solvers3D()
#加载步输出
#for opt =120:1:130
for opt =1:step_total
    A = [node[1:ncorner,:] numD[:, opt]]
    fid = open("d_data$(opt).dat", "w")
    StringVariable = "TITLE=\"2Dmodel\" VARIABLES=\"X\",\"Y\",\"Z\",\"d1\" ZONE N=$ncorner,E=$nel,F=FEPOINT,ET=TETRAHEDRON, "
    write(fid, StringVariable)
    writedlm(fid, A)
    writedlm(fid, element[:,1:4])
    close(fid)
    #Q4单元位移
    B = [node[1:ncorner,:] numD2[1:3:3*ncorner-2, opt] numD2[2:3:3*ncorner-1, opt] numD2[3:3:3*ncorner, opt]]
    fid = open("u_data$(opt).dat", "w")
    StringVariable = "TITLE=\"2Dmodel\" VARIABLES=\"X\",\"Y\",\"Z\",\"ux\",\"uy\",\"uz\" ZONE N=$ncorner,E=$nel,F=FEPOINT,ET=TETRAHEDRON, "
    write(fid, StringVariable)
    writedlm(fid, B)
    writedlm(fid, element[:,1:4])
    close(fid)
end =#
#__________________________________________________________________________________________
#Q8单元下，后4节点是加在后面的，对于node和element的编号，前面部分和Q4是相同的
#=     fid = open("err.dat", "w")
    StringVariable = "err_d&u"
    write(fid, StringVariable)
    writedlm(fid, num_err_d_u)
    close(fid) =#
#迭代步输出
#= for opt = 1:1:step_total*maxit
    A = [node num_d[:, opt]]
    fid = open("d_data$opt.dat", "w")
    StringVariable = "TITLE=\"2Dmodel\" VARIABLES=\"X\",\"Y\",\"d1\" ZONE N=$nnode,E=$nel,F=FEPOINT,ET=QUADRILATERAL, "
    write(fid, StringVariable)
    writedlm(fid, A)
    writedlm(fid, element)
    close(fid)

    B = [node num_u[1:2:2*nnode-1, opt] num_u[2:2:2*nnode, opt]]
    fid = open("u_data$opt.dat", "w")
    StringVariable = "TITLE=\"2Dmodel\" VARIABLES=\"X\",\"Y\",\"ux\",\"uy\" ZONE N=$nnode,E=$nel,F=FEPOINT,ET=QUADRILATERAL, "
    write(fid, StringVariable)
    writedlm(fid, B)
    writedlm(fid, element)
    close(fid)
end =#

#=     d_nit = [node[1:size(d1,1),:] d1]
    fid = open("d_nit$nit.dat", "w")
    StringVariable = "TITLE=\"2Dmodel\" VARIABLES=\"X\",\"Y\",\"d1\" ZONE N=$(size(d1,1)),E=$nel,F=FEPOINT,ET=QUADRILATERAL, "
    write(fid, StringVariable)
    writedlm(fid, d_nit)
    writedlm(fid, element[:,1:4])
    close(fid) 
 =#


#_________________________________________________________________________________________________________________________________
## 定义了无损状态下的刚度矩阵KK和弹性矩阵DK
#= DK = Array{Float64,2}(undef,9,4*size(element,1))
  # DK[:,Mat_ind0[:]] = kron(reshape(D1,9),ones(1,size(Mat_ind0,1)))
  # DK[:,Mat_ind12[:]] = kron(reshape(D2,9),ones(1,size(Mat_ind12,1)))
  E = Array{Float64,1}(undef,4*nel)
  # @load "D:\\Columbia_University\\precrack\\alfa=30-conf=0\\E.jld" E
  λ = Array{Float64,1}(undef,4*nel)
  μ = Array{Float64,1}(undef,4*nel)
  Kv = Array{Float64,1}(undef,4*nel)
  for iel in 1:1:size(element,1) ## weak inclusion
      E[4*(iel-1)+1:4*iel] .= E0
      λ[4*(iel-1)+1:4*iel] = E[4*(iel-1)+1:4*iel]*v/((1.0+v)*(1.0-2.0v))
      μ[4*(iel-1)+1:4*iel] = E[4*(iel-1)+1:4*iel]/(2.0*(1.0+v))
      Kv[4*(iel-1)+1:4*iel] = λ[4*(iel-1)+1:4*iel] .+ 2.0/3.0 .* μ[4*(iel-1)+1:4*iel]
      DK[:,4*(iel-1)+1:4*iel] = [λ[4*(iel-1)+1:4*iel]'.+2.0μ[4*(iel-1)+1:4*iel]'; λ[4*(iel-1)+1:4*iel]';
           zeros(1,4); λ[4*(iel-1)+1:4*iel]'; λ[4*(iel-1)+1:4*iel]'.+2.0μ[4*(iel-1)+1:4*iel]';
           zeros(1,4); zeros(1,4); zeros(1,4); μ[4*(iel-1)+1:4*iel]'] ##平面应变
      # DK[:,4*(iel-1)+1:4*iel] = kron(E[4*(iel-1)+1:4*iel]'./(1-v^2), [1; v; 0;v; 1; 0;0; 0; (1-v)/2]) ##平面应力
  end
  eK=SharedArray{Float64,2}(64,size(element,1))
  @sync @distributed for iel=1:size(element,1)
      eK[:,iel] = reshape(kron(detjacob[:,iel]',ones(8,3)).*Bu[:,8*(iel-1)+1:8*iel]'*blockdiag(sparse(reshape(DK[:,4*(iel-1)+1],3,3)),
      sparse(reshape(DK[:,4*(iel-1)+2],3,3)),sparse(reshape(DK[:,4*(iel-1)+3],3,3)),sparse(reshape(DK[:,4*(iel-1)+4],3,3)))*Bu[:,8*(iel-1)+1:8*iel],64)
  end
  KK = sparse(iKu,jKu,eK[:]) =#
#=
const Jb0 = [1/3 1/3 0.0; 1/3 1/3 0.0; 0.0 0.0 0.0]
const Kb0 = [2/3 -1/3 0.0; -1/3 2/3 0; 0 0 0.5]
Mat_set0 = [1:nel...]
DK0 = Array{Float64,2}(undef, 9, 4 * size(element, 1))
E = Array{Float64,1}(undef, 4 * nel)
λ = Array{Float64,1}(undef, 4 * nel)
μ = Array{Float64,1}(undef, 4 * nel)
Kv = Array{Float64,1}(undef, 4 * nel)
for iel in Mat_set0
    E[4*(iel-1)+1:4*iel] .= E0
    λ[4*(iel-1)+1:4*iel] .= λ0
    μ[4*(iel-1)+1:4*iel] .= μ0
    Kv[4*(iel-1)+1:4*iel] = λ[4*(iel-1)+1:4*iel] .+ 2.0 / 3.0 .* μ[4*(iel-1)+1:4*iel]
    DK0[:, 4*(iel-1)+1:4*iel] = [λ[4*(iel-1)+1:4*iel]'.+2.0μ[4*(iel-1)+1:4*iel]' λ[4*(iel-1)+1:4*iel]' zeros(1, 4)
        λ[4*(iel-1)+1:4*iel]' λ[4*(iel-1)+1:4*iel]'.+2.0μ[4*(iel-1)+1:4*iel]' zeros(1, 4)
        zeros(1, 4) zeros(1, 4) μ[4*(iel-1)+1:4*iel]'] ##平面应变
    # DK0[:,4*(iel-1)+1:4*iel] = kron(E[4*(iel-1)+1:4*iel]'./(1-v^2), [1; v; 0;v; 1; 0;0; 0; (1-v)/2]) ##平面应力
end
G = deepcopy(μ)
DK = deepcopy(DK0)
Jb = kron(vec(Jb0), ones(1, 4nel))
Kb = kron(vec(Kb0), ones(1, 4nel)) =#

#=
    Cbulk2222 = zeros(2, 2, 2, 2)
    δ = [1 0; 0 1]
    for i = 1:2
        for j = 1:2
            for k = 1:2
                for l = 1:2
                    Cbulk2222[i, j, k, l] = λ0 * δ[i, j] * δ[k, l] + μ0 * (δ[i, k] * δ[j, l] + δ[i, l] * δ[j, k])
                end # for
            end # for
        end # for
    end # for
    Cvoigt = [Cbulk2222[1,1,1,1] Cbulk2222[1,1,2,2] Cbulk2222[1,1,1,2];
              Cbulk2222[2,2,1,1] Cbulk2222[2,2,2,2] Cbulk2222[2,2,1,2];
              Cbulk2222[1,2,1,1] Cbulk2222[1,2,2,2] Cbulk2222[1,2,1,2];]#Cvoigt=Cbulk33
    Cvoigt91 = Cvoigt[:] #Cvoigt91=Cbulk_vec ,验证了voigt降阶后，列向量表示四阶矩阵 =#
