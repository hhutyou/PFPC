#
using Distributed, JLD2
addprocs(8)
@everywhere using LinearAlgebra, Distributed, SparseArrays, SharedArrays, DelimitedFiles,Pardiso,StatsBase,NearestNeighbors
include("Mesh.jl") #include functions:node, element
@everywhere include("FemBase.jl")
# using .FemBase: xdirect, ydirect, principle, invariant
include("boundary.jl")
include("shapeFunc.jl")
include("solvers.jl")
include("K_f_matrix.jl")
include("d2u.jl")
include("Hn.jl")
include("u2d.jl")
include("sigma_plus.jl")
include("sigma_minus.jl")
include("sigma_dev.jl")
include("crack_2d_display.jl")
include("regularize2d.jl")
include("SubfuncForNd.jl")
# include("solvers_initial_d.jl")
# include("integration_d.jl")
#
# elastic parameters14
const E0, v = 210.0, 0.3
const λ0, μ0 = 121.15, 80.77 ## kN/mm²
const G0, Kv0=μ0, λ0+2/3*μ0
# phase field parameters,,,
const mesh_size,ls, k = 0.00375,0.0075, 1e-16
const gc = 2.7e-3 ## kN/mm
const gc1 = gc
const Jb0 = [1/3 1/3 0.0; 1/3 1/3 0.0; 0.0 0.0 0.0]
const Kb0 = [2/3 -1/3 0.0; -1/3 2/3 0; 0 0 0.5]
## Initialization of integrative parameters
  const maxit=200
  const tol=1.0e-3
  nnode_u = size(node,1)
  nnode_d = size(union(element[:,1:4]),1)
  nel=size(element,1)
  ##
   planetype = "plane-strain"
  # const λ1, μ1 = E1*v/((1.0+v)*(1.0-2.0v)), E1/(2.0*(1.0+v))
  # const λ2, μ2 = E2*v/((1.0+v)*(1.0-2.0v)), E2/(2.0*(1.0+v))
  # const G1, Kv1=μ1, λ1+2/3*μ1
  # const G2, Kv2=μ2, λ2+2/3*μ2
  # # const D1, D2=E1/(1-v^2)*[1 v 0;v 1 0;0 0 (1-v)/2], E2/(1-v^2)*[1 v 0;v 1 0;0 0 (1-v)/2]##plane-stress
  # const D1, D2=[λ1+2.0μ1 λ1 0.0;λ1 λ1+2.0μ1 0.0;0.0 0.0 μ1], [λ2+2.0μ2 λ2 0.0;λ2 λ2+2.0μ2 0.0;0.0 0.0 μ2]  ##plane-strain
  DK0 = Array{Float64,2}(undef,9,9nel)
  # AA = Array{Float64,1}(undef,4*nel) ## frictional coefficient
  # @load "D:\\Columbia_University\\precrack\\alfa=30-conf=0\\E.jld" E
  λ = Array{Float64,1}(undef,9*nel)
  μ = Array{Float64,1}(undef,9*nel)
  Kv = Array{Float64,1}(undef,9*nel)
  for iel in 1:nel
      λ[9*(iel-1)+1:9*iel] .= λ0
      μ[9*(iel-1)+1:9*iel] .= μ0
      Kv[9*(iel-1)+1:9*iel] = λ[9*(iel-1)+1:9*iel] .+ 2.0/3.0 .* μ[9*(iel-1)+1:9*iel]
      DK0[:,9*(iel-1)+1:9*iel] = [λ[9*(iel-1)+1:9*iel]'.+2.0μ[9*(iel-1)+1:9*iel]'; λ[9*(iel-1)+1:9*iel]';
           zeros(1,9); λ[9*(iel-1)+1:9*iel]'; λ[9*(iel-1)+1:9*iel]'.+2.0μ[9*(iel-1)+1:9*iel]';
           zeros(1,9); zeros(1,9); zeros(1,9); μ[9*(iel-1)+1:9*iel]'] ##平面应变
      # DK0[:,4*(iel-1)+1:4*iel] = kron(E[4*(iel-1)+1:4*iel]'./(1-v^2), [1; v; 0;v; 1; 0;0; 0; (1-v)/2]) ##平面应力
  end
  G = deepcopy(μ)
  DK = deepcopy(DK0)
  Jb = kron(vec(Jb0),ones(1,9nel))
  Kb = kron(vec(Kb0),ones(1,9nel))
  # DK = kron(reshape(D1,9),ones(1,4*size(element,1)))
  ##🎺 DK需考虑不均质点💚
## output
u_inc1, u_inc2, step_total = 1e-4, 5e-5, Int(100+0.005/5e-5)
const numd=step_total ##output number
const aa=Int.(step_total/numd)
begin ##初始化结果储存矩阵
    numD = Array{Float64,2}(undef,nnode_d,numd); numD2 = Array{Float64,2}(undef,2*nnode_u,numd)#
    numD3 = Array{Float64,3}(undef,9,nel,numd); numD4 = Array{Int32,2}(undef,maxit,step_total+1); numD5 = Array{Float64,2}(undef,maxit,step_total+1)
    Fload1 = Array{Float64}(undef,step_total+1); Uload1 = Array{Float64}(undef,step_total+1)
    Fload2 = Array{Float64}(undef,step_total+1); Uload2 = Array{Float64}(undef,step_total+1)
    iter_storage = Array{Float64}(undef,step_total+1); time_storge = Array{Float64}(undef,step_total+1)
end
##计算过程
    ## Confinement info:注意函数Fmat_ext
        conf = 0.0
    ## Sovle
    Fload1, Uload1, Fload2, Uload2, iter_storage, time_storge, numD, numD2, numD3 = solvers(conf)
## Post-operation
    # include("out_E.jl")
    # nodeE = out_E(nel,node,element)
    # gNodeeps=pmap(out_accumulated_epsilonp, numD2[:,i] for i =1:numd)
    for opt=100:1:step_total
        # A=[node sqrt.(numD2[2:2:end,opt] .^2 .+ numD2[1:2:end,opt] .^2 )]
        A=[node[1:nnode_d,:] numD[:,opt]]
        # A=[node d1]
        # A=[node out_accumulated_epsilonp(numD5[1,:,opt])]
        #numD[:,opt] out_accumulated_epsilonp(numD2[:,opt])
        # mat=[A;element]
        fid=open("d_data$opt.dat","w")
        StringVariable="TITLE=\"2Dmodel\" VARIABLES=\"X\",\"Y\",\"d1\" ZONE N=$nnode_d,E=$nel,F=FEPOINT,ET=QUADRILATERAL, "
        write(fid,StringVariable)
        # m,n=size(mat)
        writedlm(fid,A)
        writedlm(fid,element[:,1:4])
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
    # @save pwd()*"\\numD4.jld" numD4
    # @save pwd()*"\\numD5.jld" numD5
    # @save pwd()*"\\numD6.jld" numD6
    # @save pwd()*"\\numD7.jld" numD7
    # @save pwd()*"\\numD8.jld" numD8
    # @save "D:\\Columbia_University\\precrack\\alfa=30-conf=0\\E.jld" E
