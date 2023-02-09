#abaqus中的裂纹路径单元号 EleCrack（裂纹预设三单元宽，取中间一单元宽的作为EleCrack）
#C_update.jl中 切换裂纹识别方式crack_orientation（Fan）/crack_regularize2d（RidgeRegression） 

#%%
using Distributed,Plots,StatsBase,NearestNeighbors
addprocs(8-nprocs())
println("Running ",nprocs()," processes")
@everywhere using LinearAlgebra, SparseArrays, SharedArrays, DelimitedFiles, Pardiso
include("Mesh.jl")
# loading&iterations
const step_total, u_inc = 10, -0.01
const maxit = 3
const maxit_u = 3
const tol = 1e-3
const tol_u = 1e-3
# phase field parameters
const ls,mesh_size, kk = 0.016,0.004, 1e-2
const gc = 5e4 #N/m

# constitutive parameters    单位m,Pa,N
const inter_detec = 0.9
const μ_fric = 0.1 #friction coefficient
const threshold = 0.99 #损伤阈值
const E0, v = 1e10, 0.3
const λ0, μ0 = 5.77e9, 3.85e9 #λ0=E0*v/((1+v)*(1-2*v)),μ0=E0/(2*(1+v))
const G0, Kv0 = 3.85e9, 8.33e9  #G0=μ0,Kv0=E0/(3*(1-2*v))
const planetype = "plane-strain"
const Cbulk_vec = [λ0 + 2 * G0; λ0; 0; λ0; λ0 + 2 * G0; 0; 0; 0; G0]
const Cbulk33 = [λ0 + 2*G0 λ0 0; λ0 λ0+2*G0 0; 0 0 G0]  #用γ就是G0 #plain-strain
const δ = [1 0; 0 1]
@everywhere include("FemBase.jl")
include("boundary.jl")
include("shapeFuncQ8.jl")
include("Hn.jl")
include("d_calcu.jl")
include("C_update.jl")
include("K_f_matrix.jl")
include("u_calcu.jl")
include("solversInternal.jl")
#拟合裂纹路径
include("crack_regularize2d.jl")
include("regularize2d.jl")
include("SubfuncForNd.jl")
include("findPointNormals2d.jl")

include("crack_orientation.jl")
const numd = step_total #所有步都存
const aa = 1 #间隔1 

numD = Array{Float64,2}(undef, ncorner, step_total)#d
numD2 = Array{Float64,2}(undef, 2*ncorner, step_total)#u
numD3 = Array{Float64,2}(undef, nel, step_total)#σ 单元应力gauss点求平均

#输出裂纹界面单元的正应力、切应力（倍数关系）
EleCrack = [    3,   41,   42,   44,   46,   47,   49,   51,   52,   54,   56,   57,   59,   61,   62,   64,
    67,   68,   70,   72,   73,   75,   77,   80,   82,   83,   85,   92,  183, 2665, 2853, 2926,
    3046, 3503, 3547, 5662, 5664, 5667, 5668, 5670, 5673, 5677, 5680, 5682, 5684, 5686, 5687, 5691,
    5692, 5695, 5698, 5700, 5701, 5703, 5707, 5708, 5710, 5712, 5715, 5719, 5722, 5723, 5725, 5726,
    5730, 5734, 5735, 5737, 5741, 5746, 5749, 5751, 5753, 5754, 5757, 5762, 5764, 5768, 5770, 5774,
    5775, 5776, 5778, 5779, 5780, 5783, 5784, 5785, 5790, 5793, 5794, 5796, 5798, 5799, 5803, 5804,
    5805, 5806, 5808, 5809, 5810, 5811, 5812, 5813, 5815, 5998, 6004, 6008, 6013, 6017, 6024, 6031,
    6036, 6042, 6048, 6049, 6052, 6058, 6111, 6119, 6125, 6128, 6130, 6141, 6146, 6153, 6154, 6164,
    6169, 6170, 6182, 6184, 6190, 6192, 6202, 6207, 6213, 6216, 6218, 6219, 6225]
#

#%%开始计算
solversInternal(u_inc)
#solvers(u_inc)

#%%
#= fid = open("d_data0.dat", "w")
StringVariable = "TITLE=\"2Dmodel\" VARIABLES=\"X\",\"Y\",\"d1\" ZONE N=$ncorner,E=$nel,F=FEPOINT,ET=QUADRILATERAL, "
write(fid, StringVariable)
writedlm(fid, [node[1:ncorner,:] d1])
writedlm(fid, element[:,1:4])
close(fid) =#
#%%
#加载步输出
for opt =1:step_total
    A = [node[1:ncorner,:] numD[:, opt]]
    fid = open("d_data$(opt).dat", "w")
    StringVariable = "TITLE=\"2Dmodel\" VARIABLES=\"X\",\"Y\",\"d\" ZONE N=$ncorner,E=$nel,F=FEPOINT,ET=QUADRILATERAL, "
    write(fid, StringVariable)
    writedlm(fid, A)
    writedlm(fid, element[:,1:4])
    close(fid)
    #Q4单元位移 
    B = [node[1:ncorner,:] numD2[1:2:2*ncorner-1, opt] numD2[2:2:2*ncorner, opt]]
    fid = open("u_data$(opt).dat", "w")
    StringVariable = "TITLE=\"2Dmodel\" VARIABLES=\"X\",\"Y\",\"Ux\",\"Uy\" ZONE N=$ncorner,E=$nel,F=FEPOINT,ET=QUADRILATERAL, "
    write(fid, StringVariable)
    writedlm(fid, B)
    writedlm(fid, element[:,1:4])
    close(fid)
end
#Q8单元下，后4节点是加在后面的，对于node和element的编号，前面部分和Q4是相同的



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
