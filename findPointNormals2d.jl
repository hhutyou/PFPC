#%%输入二维点信息
#= xGrid2 = -0.5:0.05:3.6
yGrid2 = [0.9029062450380063
    0.9103610126286659
    0.9178157802193252
    0.9252705478099843
    0.9327253154006434
    0.9401800829913027
    0.9476348505819622
    0.9550896181726216
    0.9625443857632813
    0.969999153353941
    0.9774539209446009
    0.984908688535261
    0.9925250260891527
    1.0004645035695074
    1.0090139202925335
    1.0184600755744395
    1.029168497298385
    1.0415047133475288
    1.0558402103289655
    1.07254647484979
    1.0918814792251375
    1.1141031957701442
    1.1391694652763291
    1.1669370620563604
    1.1972627604229071
    1.230003334688639
    1.2650155591662255
    1.3021562081683362
    1.3412820560076404
    1.3822498769968077
    1.4249164454485077
    1.4691385356754103
    1.5147729219901869
    1.561676378705508
    1.609599814255683
    1.6582941370750226
    1.7075102555978372
    1.7569990782584375
    1.806511513491134
    1.8557984697302377
    1.9046108554100591
    1.9526995789649095
    1.9998155488291023
    2.04570967343695
    2.0901328612227674
    2.1328360206208674
    2.1735700600655643
    2.2120858879911727
    2.248134412832006
    2.2814665430223786
    2.311833186996605
    2.3389852531890014
    2.3626736500338827
    2.3826492859655657
    2.3986630694183657
    2.4104659088265996
    2.4178087126245846
    2.420442389246639
    2.4181178471270792
    2.410585994700225
    2.3975977404003945
    2.3789039926619036
    2.3542556599190703
    2.323403650606211
    2.2871433078733308
    2.2462699748704336
    2.2015789947475226
    2.1538657106546
    2.103925465741667
    2.0525536031587253
    2.000570917036579
    1.9486264073856328
    1.896681897734687
    1.8447373880837417
    1.7927928784327964
    1.7408483687818517
    1.6889038591309071
    1.6369593494799626
    1.585014839829018
    1.5330703301780735
    1.4811258205271292
    1.4291813108761846
    1.3772368012252398]
##
points = [xGrid2 yGrid2]  =# # 83 * 2
#%%

function findPointNormals2d(points, numNeighbours::Int, viewPoint, dirLargest)
    # instructions
     #FINDPOINTNORMALS Estimates the normals of a sparse set of n 3d points by
     # using a set of the closest neighbours to approximate a plane.
     #
     #   Required Inputs:
     #   points- nx3 set of 3d points [x,y,z]
     #
     #   Optional Inputs: (will give default values on empty array [])
     #   numNeighbours- number of neighbouring points to use in plane fitting
     #       (default 9)
     #   viewPoint- location all normals will point towards [default [0,0,0]]
     #   dirLargest- use only the largest component of the normal in determining
     #       its direction wrt the viewPoint (generally provides a more stable
     #       estimation of planes near the viewPoint, default true)
     #
     #   Outputs:
     #   normals- nx3 set of normals [nx,ny,nz]
     #   curvature- nx1 set giving the curvature
     #
     #   References-
     #   The implementation closely follows the method given at
     #   http://pointclouds.org/documentation/tutorials/normal_estimation.php
     #   This code was used in generating the results for the journal paper
     #   Multi-modal sensor calibration using a gradient orientation measure 
     #   http://www.zjtaylor.com/welcome/download_pdf?pdf=JFR2013.pdf
     #
     #   This code was written by Zachary Taylor
     #   zacharyjeremytaylor@gmail.com
     #   http://www.zjtaylor.com
    ## check inputs
       # numNeighbours默认为9,太大则long run times,large ram usage and poor results near edges
       #= if (isempty(numNeighbours))
           numNeighbours = 9
       else
           validateattributes[numNeighbours, ("numeric'},{'scalar','positive")]
           if (numNeighbours .> 100)
               warning("#i neighbouring points will be used in plane"
                   " estimation; expect long run times; large ram usage and"
                   " poor results near edges",numNeighbours)
           end
       end =#
      
      #= if isempty(viewPoint)==1
          viewPoint = [0,0,0]
      else
          validateattributes[viewPoint, ("numeric'},{'size",[1,3])]
      end =#
      
      #= if (nargin .< 4)
          dirLargest = []
      end =#
      #= if isempty(dirLargest)==1
          dirLargest = true
      else
          validateattributes[dirLargest, ("logical'},{'scalar")]
      end =#
    
    ## setup
    
    #ensure inputs of correct type()
    #= points = double(points)
    viewPoint = double(viewPoint) =#

    # Create kdtrees
    kdtreeobj = KDTree(points';leafsize = 10)
    #numNeighbours = 9 #结束需删除！！！
    idxs, dists = knn(kdtreeobj, points', numNeighbours+1, true)
    
    #get nearest neighbours
    #n = knnsearch[kdtreeobj,points,'k',(numNeighbours+1)]
    
    #remove self
    idxs = collect(reduce(hcat,idxs)')
    # or vcat(x’…)
    idxs = idxs[:,2:end]
    #find difference in position from neighbouring points
    p = repeat(points[:,1:2],numNeighbours,1) - points[idxs[:],1:2]
    p = reshape(p, size(points,1),numNeighbours,2)
    
    #calculate values for covariance matrix
    C = zeros(size(points,1),3)
    C[:,1] = sum(p[:,:,1].*p[:,:,1],dims=2)
    C[:,2] = sum(p[:,:,1].*p[:,:,2],dims=2)
    C[:,3] = sum(p[:,:,2].*p[:,:,2],dims=2)

    C = C ./ numNeighbours
    
    ## normals & curvature calculation 主元分析
    
    normals = zeros(size(points))
    curvature = zeros(size(points,1),1)
    for i = 1:(size(points,1))  #parallel?
        
        #form covariance matrix
        Cmat = [C[i,1] C[i,2] 
                C[i,2] C[i,3] ]  
        
        #get eigen values & vectors
        d,v = eigen(Cmat)
        lambda = minimum(d)
        k = 1 #???
    
        #store normals
        normals[i,:] = v[:,k]'
        
        #store curvature
        curvature[i] = lambda / sum(d)
    end
    
    ## flipping normals
    #viewPoint = [0 0]
    #ensure normals point towards viewPoint
    points = points - repeat(viewPoint,size(points,1),1)
    if dirLargest == true
        idx = findall(abs.(normals).==maximum(abs.(normals),dims = 2))
        #该到这里！！！
        idx = map(i->i[2], idx)
        idx = (1:size(normals,1)) + (idx.-1)*size(normals,1)
        dir = normals[idx].*points[idx] .> 0
    else
        dir = sum(normals.*points,dims=2) .> 0
    end
    
    normals[dir,:] = -normals[dir,:]
    
    return  normals #, curvature 
    
    end