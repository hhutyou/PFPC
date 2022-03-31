#
using Distributed, JLD2
addprocs(4-nprocs())
@everywhere using LinearAlgebra, Distributed, SparseArrays, SharedArrays, DelimitedFiles
include("Mesh.jl") #include functions:node, element
@everywhere include("FemBase.jl")
# using .FemBase: xdirect, ydirect, principle, invariant
include("boundary.jl")
include("ShapeFuncHexLinear.jl")
include("solvers.jl")
include("K_f_matrix3D.jl")
include("d2u.jl")
include("Hn.jl")
include("u2d.jl")
include("sigma_plus.jl")
include("sigma_minus.jl")
include("sigma_dev.jl")
# include("solvers_initial_d.jl")
# include("integration_d.jl")
#
# elastic parameters14
const E0, v = 210.0, 0.3
const λ0, μ0 = 121.15, 80.77 ## kN/mm²
const G0, Kv0=μ0, λ0+2/3*μ0
# phase field parameters,,,
const ls, k = 0.0075, 1e-16
const gc = 2.7e-3 ## kN/mm
const gc1 = gc
const Jb0 = 1/3 * [1.0  0.0  0.0  0.0  0.0  0.0;
            0.0  1.0  0.0  0.0  0.0  0.0;
            0.0  0.0  1.0  0.0  0.0  0.0;
            0.0  0.0  0.0  1.0  0.0  0.0;
            0.0  0.0  0.0  0.0  1.0  0.0;
            0.0  0.0  0.0  0.0  0.0  1.0]
const Kb0 = [2/3  -1/3  -1/3  0.0  0.0  0.0;
            -1/3   2/3  -1/3  0.0  0.0  0.0;
            -1/3  -1/3   2/3  0.0  0.0  0.0;
            0.0    0.0   0.0  0.5  0.0  0.0;
            0.0    0.0   0.0  0.0  0.5  0.0;
            0.0    0.0   0.0  0.0  0.0  0.5]
## Initialization of integrative parameters
  const maxit=30
  const reltol=1.0e-3
  const abstol_u = 1e-4
  const abstol_d = 1e-4
  nnode_u = size(node,1)
  nnode_d = size(node,1)
  nel=size(element,1)
  ##
   planetype = "C3D8"
  # const λ1, μ1 = E1*v/((1.0+v)*(1.0-2.0v)), E1/(2.0*(1.0+v))
  # const λ2, μ2 = E2*v/((1.0+v)*(1.0-2.0v)), E2/(2.0*(1.0+v))
  # const G1, Kv1=μ1, λ1+2/3*μ1
  # const G2, Kv2=μ2, λ2+2/3*μ2
  # # const D1, D2=E1/(1-v^2)*[1 v 0;v 1 0;0 0 (1-v)/2], E2/(1-v^2)*[1 v 0;v 1 0;0 0 (1-v)/2]##plane-stress
  # const D1, D2=[λ1+2.0μ1 λ1 0.0;λ1 λ1+2.0μ1 0.0;0.0 0.0 μ1], [λ2+2.0μ2 λ2 0.0;λ2 λ2+2.0μ2 0.0;0.0 0.0 μ2]  ##plane-strain
  DK0 = Array{Float64,2}(undef,36,8nel)
  # AA = Array{Float64,1}(undef,4*nel) ## frictional coefficient
  # @load "D:\\Columbia_University\\precrack\\alfa=30-conf=0\\E.jld" E
  λ = Array{Float64,1}(undef,8*nel)
  μ = Array{Float64,1}(undef,8*nel)
  Kv = Array{Float64,1}(undef,8*nel)
  for iel in 1:nel
      λ[8*(iel-1)+1:8*iel] .= λ0
      μ[8*(iel-1)+1:8*iel] .= μ0
      Kv[8*(iel-1)+1:8*iel] = λ[8*(iel-1)+1:8*iel] .+ 2.0/3.0 .* μ[8*(iel-1)+1:8*iel]
      DK0[:,8*(iel-1)+1:8*iel] = [λ[8*(iel-1)+1:8*iel]'.+2.0μ[8*(iel-1)+1:8*iel]'; λ[8*(iel-1)+1:8*iel]'; λ[8*(iel-1)+1:8*iel]'; zeros(1,8); zeros(1,8); zeros(1,8);
                                  λ[8*(iel-1)+1:8*iel]'; λ[8*(iel-1)+1:8*iel]'.+2.0μ[8*(iel-1)+1:8*iel]'; λ[8*(iel-1)+1:8*iel]'; zeros(1,8); zeros(1,8); zeros(1,8);
                                  λ[8*(iel-1)+1:8*iel]'; λ[8*(iel-1)+1:8*iel]'; λ[8*(iel-1)+1:8*iel]'.+2.0μ[8*(iel-1)+1:8*iel]'; zeros(1,8); zeros(1,8); zeros(1,8);
                                  zeros(1,8); zeros(1,8); zeros(1,8); μ[8*(iel-1)+1:8*iel]'; zeros(1,8); zeros(1,8);
                                  zeros(1,8); zeros(1,8); zeros(1,8); zeros(1,8); μ[8*(iel-1)+1:8*iel]'; zeros(1,8);
                                  zeros(1,8); zeros(1,8); zeros(1,8); μ[8*(iel-1)+1:8*iel]'; zeros(1,8); μ[8*(iel-1)+1:8*iel]'] ##三维
  end
  G = deepcopy(μ)
  DK = deepcopy(DK0)
  Jb = repeat(vec(Jb0),1,8nel)
  Kb = repeat(vec(Kb0),1,8nel)
  # DK = kron(reshape(D1,8),ones(1,4*size(element,1)))
  ##🎺 DK需考虑不均质点💚
## output
u_inc1, u_inc2, step_total = 1e-4, 5e-5, Int(100+0.005/5e-5)
const numd=step_total ##output number
const aa=Int.(step_total/numd)
##计算及输出过程
begin
    solvers()
end

