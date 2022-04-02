
function regularize2d(x, y, xGrid, smoothness, interpMethod, solver, maxIterations, solverTolerance)
    # instructions     
        # regularizeNd  Fits a nD lookup table with smoothness to scattered data.
        #
        #   yGrid = regularizeNd(x, y, xGrid)
        #   yGrid = regularizeNd(x, y, xGrid, smoothness)
        #   yGrid = regularizeNd(x, y, xGrid, smoothness, interpMethod)
        #   yGrid = regularizeNd(x, y, xGrid, smoothness, interpMethod, solver)
        #   yGrid = regularizeNd(x, y, xGrid, smoothness, interpMethod, solver, maxIterations)
        #   yGrid = regularizeNd(x, y, xGrid, smoothness, interpMethod, solver, maxIterations, solverTolerance)
        #
        ## Inputs
        #      x - column vector | matrix of column vectors; containing scattered
        #          data. Each row contains one point. Each column corresponds to a
        #          dimension.
        #
        #      y - vector containing the corresponds values to x. y has the same
        #          number of rows as x.
        #
        #  xGrid - cell array containing vectors defining the nodes in the grid in
        #          each dimension. xGrid[1] corresponds with x[:,1] for instance.
        #          Unequal spacing in the grid vectors is allowed. The grid vectors
        #          must completely span x. For instance the values of x[:,1] must
        #          be within the bounds of xGrid[1]. If xGrid does not span x
        #          an error is thrown. 
        #
        #  smoothness - scalar | vector. - The numerical "measure" of what we want
        #          to achieve along an axis/dimension; regardless of the
        #          resolution; the aspect ratio between axes; | the scale of the
        #          overall problem. The ratio of smoothness to fidelity of the
        #          output surface; i.e. ratio of smoothness to "goodness of
        #          fit." This must be a positive real number. If it is a vector; it
        #          must have same number of elements as columns in x.
        #
        #          A smoothness of 1 gives equal weight to fidelity [goodness of fit]
        #          & smoothness of the output surface.  This results in noticeable
        #          smoothing.  If your input data has little | no noise; use
        #          0.01 to give smoothness 1# as much weight as goodness of fit.
        #          0.1 applies a little bit of smoothing to the output surface.
        #
        #          If this parameter is a vector; then it defines the relative
        #          smoothing to be associated with each axis/dimension. This allows
        #          the user to apply a different amount of smoothing in the each
        #          axis/dimension.
        #
        #          DEFAULT: 0.01
        #
        #   interpMethod - character; denotes the interpolation scheme used
        #          to interpolate the data.
        #
        #          Even though there is a computational complexity difference between
        #          linear; nearest; & cubic interpolation methods; the
        #          interpolation method is not the dominant factor in the
        #          calculation time in regularizeNd. The dominant factor in
        #          calculation time is the size of the grid & the solver used. So
        #          in general; do not choose your interpolation method based on
        #          computational complexity. Choose your interpolation method because
        #          of accuracy & shape that you are looking to obtain.
        #
        #          "linear" - Uses linear interpolation within the grid. linear
        #                     interpolation requires that extrema occur at the grid()
        #                     points. linear should be smoother than nearest for
        #                     the same grid. As the number of dimension grows
        #                     the number of grid points used to interpolate at a
        #                     query point grows with 2^nDimensions. i.e. 2d needs 4
        #                     points; 3d needs 8 points; 4d needs 16 points per
        #                     query point. In general; linear can use smaller
        #                     smoothness values than cubic & still be well
        #                     conditioned.
        #
        #          "nearest" - Nearest neighbor interpolation. Nearest should
        #                      be the least complex but least smooth.
        #
        #          "cubic" - Uses Lagrange cubic interpolation. Cubic interpolation
        #                    allows extrema to occur at other locations besides the
        #                    grid points. Cubic should provide the most flexible
        #                    relationship for a given xGrid. As the number of
        #                    dimension grows; the number of grid points used to
        #                    interpolate at a query point grows with 4^nDimensions.
        #                    i.e. 2d needs 16 points; 3d needs 64 points; 4d needs
        #                    256 points per query point. cubic has good properties
        #                    of accuracy & smoothness but is the most complex()
        #                    interpMethod to calculate.
        #
        #          DEFAULT: "linear"
        #
        #
        #   solver - string that denotes the solver used for the
        #            resulting linear system. The default is most often the best
        #            choice.
        #
        #          What solver should you use? The short answer is use "normal" as
        #          a first guess. '\' may be best numerically for most smoothness
        #          parameters & high extents of extrapolation. If you receive
        #          rank deficiency warnings with "normal'; try the '\" solver.
        #          Otherwise; use the "normal" solver because it is usually faster
        #          than the '\' solver.
        #
        #          The larger the numbers of grid points; the larger the solve time.
        #          Since the equations generated tends to be well conditioned; the
        #          "normal' solver is  a good choice. Beware using 'normal" when a
        #          small smoothing parameter is used; since this will make the
        #          equations less well conditioned. The "normal" solver for large
        #          grids is 3x faster than the '\'.
        #
        #          Use the "pcg'; 'symmlq'; | 'lsqr' solver when the 'normal" &
        #          "\' fail. Out of memory errors with 'normal' | '\" are reason to
        #          try the iterative solvers. These errors are rare however they
        #          happen. Start with the "pcg' solver. Then 'symmlq". Finally try()
        #          "lsqr' solver. The 'lsqr" solver is usually slow compared to the
        #          "pcg' & 'symmlq" solver.
        #
        #          "\' - uses matlab"s backslash operator to solve the sparse()
        #                system.
        #
        #          "lsqr" - Uses the MATLAB lsqr solver. This solver is not
        #                   recommended. Try "pcg' | 'symmlq" first & use
        #                   "lsqr" as a last resort. Experiments have shown that
        #                   "pcg' and 'symmlq" solvers are faster & just as
        #                   accurate as "lsqr" for the matrices generated by
        #                   regularizeNd. The same preconditioner as
        #                   the "pcg" solver is used.
        #
        #          "normal" - Constructs the normal equation & solves.
        #                     x = (A"A)\(A"*y). From testing, this seems to be a well
        #                     conditioned & faster way to solve this type of
        #                     equation system than backslash x = A\y. Testing shows
        #                     that the normal equation is 3x faster than the '\'
        #                     solver for this type of problem. A'*A preserves the
        #                     sparsity & is symmetric positive definite. Often
        #                     A'*A will have less nonzero elements than A. i.e.
        #                     nnz(A'*A) .< nnz(A).
        #                 
        #          "pcg" - Calls the MATLAB pcg iterative solver that solves the
        #                  normal equation, (A"A)*x = A"*y, for x. Use this solver
        #                  first when "normal' & '\' fail. The 'pcg" solver tries
        #                  to generate the Incomplete Cholesky Factorization
        #                  (ichol) as a preconditioner. If Incomplete Cholesky
        #                  Factorization fails; then diagonal compensation is()
        #                  added. There may be a case where the preconditioner just
        #                  cannot be calculated & thus no preconditioner is used.
        #
        #          "symmlq" - Calls the MATLAB symlq iterative solver that solves
        #                     the normal equation, (A"A)*x = A"*y, for x. Use this
        #                     solver if "pcg' has issues. 'symmlq" uses the same
        #                     preconditioner as "pcg".
        #
        #          DEFAULT: "normal"
        #
        #
        #   maxIterations - Only used if the solver is set to the iterative
        #                   solvers; "lsqr'; 'pcg'; | 'symmlq". Reducing this will
        #                   speed up the solver at the cost of accuracy. Increasing
        #                   it will increase accuracy at the cost of time. The
        #                   default value is the smaller of 100;000 & the number
        #                   of nodes in the grid.
        #
        #          DEFAULT: min(1e5,  nTotalGridPoints)
        #
        #
        #   solverTolerance - Only used if the solver is set to the iterative
        #                     solvers; "lsqr'; 'pcg'; | 'symmlq". The
        #                     solverTolerance is used with "lsqr'; 'pcg"; |
        #                     "symmlq". Smaller increases accuracy & reduces
        #                     speed. Larger decreases accuracy & increases speed.
        #
        #          DEFAULT: 1e-11*abs(maximum(y) - minimum(y))
        #
        #
        ## Output
        #  yGrid   - array containing the fitted surface | hypersurface
        #            corresponding to the grid points xGrid. yGrid is in the ndgrid()
        #            format. In 2d; ndgrid format is the transpose of meshgrid()
        #            format.
        #
        ## Description
        # regularizeNd answers the question what is the best possible lookup table
        # that the scattered data input x & output y in the least squares sense
        # with smoothing? regularizeNd is meant to calculate a smooth lookup table
        # given n-D scattered data. regularizeNd supports extrapolation from a
        # scattered data set as well.
        #
        # The calculated lookup table yGrid is meant to be used with
        # griddedInterpolant class with the conservative memory form. Call
        # griddedInterpolant like F = griddedInterpolant(xGrid, yGrid).
        # 
        # Desirable properties of regularizeNd()
        #     - Calculates a relationship between the input x & the output y
        #       without definition of the functional form of x to y.
        #     - Often the fit is superior to polynomial type fitting without 
        #       the wiggles.
        #     - Extrapolation is possible from a scattered data set. 
        #     - After creating the lookup table yGrid & using it with
        #       griddedInterpolant; as the query point moves away from the
        #       scattered data; the relationship between the input x & output y
        #       becomes more linear because of the smoothness equations & no
        #       nearby fidelity equations. The linear relationship is a good
        #       choice when the relationship between x & y is unknown in
        #       extrapolation.
        #     - regularizeNd can handle 1D; 2D; nD input data to 1D output data.
        #        RegularizeData3D and gridfit can only handle 2D input & 1D out
        #       (total 3D). 
        #     - regularizeNd can handle setting the smoothness to 0 in any()
        #        axis/dimension. This means no smoothing is applied in a particular
        #        axis/dimension & the data is just a least squares fit of a lookup
        #        table in that axis/dimension.
        #
        #  For an introduction on how regularization works; start here:
        #  https://mathformeremortals.wordpress.com/2013/01/29/introduction-to-regularizing-with-2d-data-part-1-of-3/
        #
        ## Acknowledgement
        # Special thanks to Peter Goldstein; author of RegularizeData3D; for his
        # coaching & help through writing regularizeNd.
        #
        ## Example
        #
        # # setup some input points; output points; & noise
        # x = 0.5:0.1:4.5
        # y = 0.5:0.1:5.5
        # [xx,yy] = ndgrid(x,y)
        # z = tanh(xx-3).*sin(2*pi/6*yy)
        # noise = (rand(size(xx))-0.5).*xx.*yy/30
        # zNoise = z + noise
        # 
        # # setup the grid for lookup table
        # xGrid = range(0,6,length=210)
        # yGrid = range(0,6.6,length=195)
        # gridPoints = (xGrid, yGrid)
        # 
        # # setup some difference in scale between the different dimensions/axes()
        # xScale = 100
        # x = xScale*x
        # xx=xScale*xx
        # xGrid = xScale*xGrid
        # gridPoints[1] = xGrid; 
        #
        # # smoothness parameter. i.e. fit is weighted 1000 times greater than
        # # smoothness.
        # smoothness = 0.001
        # 
        # # regularize
        # zGrid = regularizeNd([xx[:], yy[:]], zNoise[:], gridPoints, smoothness)
        # # Note this s the same as 
        # # zGrid = regularizeNd([xx[:], yy[:]], zNoise[:], gridPoints, smoothness, "linear', 'normal")
        #
        # # create girrdedInterpolant function
        # F = griddedInterpolant(gridPoints, zGrid, "linear")
        # 
        # # plot & compare
        # surf(x,y,z", 'FaceColor', 'g")
        # hold all()
        # surf(x,y,zNoise",'FaceColor', 'm")
        # surf(xGrid, yGrid, zGrid", 'FaceColor', 'r")
        # xlabel('x')
        # ylabel('y')
        # zlabel('z')
        # legend(("Exact', 'Noisy', 'regularizeNd'),'location', 'best")
        #
    
        # Author[s]: Jason Nicholson
    
    ## 
    
    # Set default smoothness | check smoothnessb 不需要
    
    # check that the grid is of the right type()
    # xGrid must be a cell array.
    
    # calculate the number of dimension  输入大数组x的维数（1D/2D）
    nDimensions = size(x,2) 
    
    # check for the matching dimensionality
    if nDimensions != length(xGrid)
        error("Dimensionality mismatch. input vars 'x & xGrid' ")
    end
    # Check if smoothness is a scalar. If it is; convert it to a vector
    if length(smoothness)==1
        smoothness = ones(nDimensions,1).*smoothness
    end
    
    # Set default interp method("linear") | check method 不需要，每个参数都输入，不弄类struct
    
    # Set default solver("normal") | check the solver
    
    # Check the grid vectors is a cell array
    if isa(xGrid,Tuple) != 1
        error("xGrid is not a tuple")
    end
        
    #dimensions&number of grid points！！！
    if nDimensions == 2
        xGrid = (collect(xGrid[1]),collect(xGrid[2]))
        # calculate the number of points in each dimension of the grid()
        nGrid = [length(xGrid[1]) length(xGrid[2])]
        nTotalGridPoints = prod(nGrid)
    elseif nDimensions == 1
        #xGrid = (collect(xGrid[1]))
        nGrid = [length(xGrid[1])]
        nTotalGridPoints = prod(nGrid,dims=1)[1]
    end
    
    # check maxIterations if the solver is iterative
    maxIterations = min(100000, nTotalGridPoints)
    # check solverTolerance if the solver is iterative
    solverTolerance = abs(maximum(y)-minimum(y))*1e-11
    # maxIterations & solverTolerance Only used if the solver is set to the iterative solvers, 'lsqr', 'pcg', or 'symmlq'. 
    
    # Check y rows matches the number in x
    nScatteredPoints = size(x,1)
    if nScatteredPoints != size(y, 1)
        error("input x and output y must have same number of rows")
    end
    
    # Check input points are within min & max of grid.输入变量必须在grid范围内！可以是闭区间，此处不检查了
    if nDimensions == 2
        xGridMin = [minimum(xGrid[1]) minimum(xGrid[2])]
        xGridMax = [maximum(xGrid[1]) maximum(xGrid[2])]
    elseif nDimensions == 1
        xGridMin = [minimum(xGrid[1]) ]
        xGridMax = [maximum(xGrid[1]) ]
    end
    #assert[all(all(bsxfun[@ge, x, xGridMin])) & all(all(bsxfun[@le, x, xGridMax])), "All #s points must be within the range of the grid vectors", getname(x)]
    
    
    
    # calculate the difference between grid points for each dimension 计算网格差值，检查单调性的
    #= if nDimensions == 2
        dx = [diff(xGrid[1]) diff(xGrid[2])]
    elseif nDimensions == 1
        dx = [diff(xGrid[1]) ]
    end =#
    # Check for monotonic increasing grid points in each dimension
    #monotonic increasing grid points 须 >0! 单调递增的，不可相等，此处不检查了
    
    
    # Check that there are enough points to form an output surface. Linear &
    # nearest interpolation types require 3 points in each output grid
    # dimension because of the numerical 2nd derivative needs three points.
    # Cubic interpolation requires 4 points per dimension.
    # 判断各个维度上插值点数量是否足够
    if interpMethod == "linear"
        minGridVectorLength = 3
    elseif interpMethod == "nearest"
        minGridVectorLength = 3
    elseif interpMethod == "cubic"
        minGridVectorLength = 4
    else
        error("wrong interpMethod.")
    end
    if all(nGrid .>= minGridVectorLength) != 1
        error("Not enough grid points in each dimension")
    end 
        
            
    
    ## Calculate Fidelity Equations  #tuple只看二维！！！ 
    
    if interpMethod == "nearest" # nearest neighbor interpolation in a cell()  /ok
        # Preallocate before loop
        xWeightIndex = ((zeros(Int,size(x,1),1)),(zeros(Int,size(x,1),1)))
       
        # a = ([],[])#Julia中元组不可变！所以设置为array而不是tuple
        for iDimension = 1:nDimensions
            # Find cell index
            # determine the cell the x-points lie in the xGrid
            # loop over the dimensions/columns; calculating cell index
            h1 = StatsBase.fit(Histogram,x[:,iDimension],xGrid[iDimension],closed=:left)
            xIndex = StatsBase.binindex.(Ref(h1), x[:,iDimension]) 
            # any point falling at the last node is taken to be
            # inside the last cell in x[:,1]|x[:,2]  #处理最后一个点
            k=findall(indx.== length(xGrid[iDimension]))
            xIndex[k] = xIndex[k] .-1
            # Calculate the cell fraction. This corresponds to a value between 0 & 1.
            # 0 corresponds to the beginning of the cell. 1 corresponds to the end of
            # the cell. The min & max functions help ensure the output is always
            # between 0 & 1.
            cellFraction = min.(1, max.(0.0,(x[:,iDimension] .- xGrid[iDimension][xIndex])./dx[:,iDimension][xIndex] )  )
            # calculate the index of nearest point
            xWeightIndex[iDimension] .= round.(cellFraction) .+ xIndex
        end
        # clean up a little释放不需要的变量     cellFraction/xIndex暂时先不clear
        #clear(getname(cellFraction), getname(xIndex))
        #cellFraction = nothing;xIndex = nothing
        
        # calculate linear index  #这是子函数了
        xWeightIndex = subscript2index(nGrid, xWeightIndex)
    
        # the weight for nearest interpolation is just 1
        weight  = 1
        
        # Form the sparse Afidelity matrix for fidelity equations
        Afidelity = sparse(1:nScatteredPoints, vec(xWeightIndex), weight, nScatteredPoints, nTotalGridPoints)
        
    elseif interpMethod == "linear"  # linear interpolation /ok
        # This will be needed below
        # Each cell has 2^nDimension nodes. The local dimension index label is 1 | 2 for each dimension. For instance; cells in 2d
        # have 4 nodes with the following indexes:
        # node label  =  1  2  3  4
        # index label = [1, 1, 2, 2
        #                1, 2, 1, 2]
        # Said in words; node 1 is one; one. node 2 is one; two. node
        # three is two; one. node 4 is two; two.
        localCellIndex = [1 1 2 2;1 2 1 2]
    
        # preallocate
        weight = ones(nScatteredPoints, 2^nDimensions)
        ##  先看2D的！！！ 暂时不考虑1D，如果1d需要另外写！
        xWeightIndex = ((zeros(Int,size(x,1),4)),(zeros(Int,size(x,1),4)))
    
        # loop over dimensions calculating subscript index in each dimension for
        # scattered points.
        
        for iDimension = 1:nDimensions
            # Find cell index
            # determine the cell the x-points lie in the xGrid
            # loop over the dimensions/columns; calculating cell index
            h1 = StatsBase.fit(Histogram,x[:,iDimension],xGrid[iDimension],closed=:left)
            xIndex = StatsBase.binindex.(Ref(h1), x[:,iDimension]) 
            # Calculate the cell fraction. This corresponds to a value between 0 & 1.
            # 0 corresponds to the beginning of the cell. 1 corresponds to the end of
            # the cell. The min & max functions help ensure the output is always
            # between 0 & 1.
            cellFraction = min.(1, max.(0.0,(x[:,iDimension] .- xGrid[iDimension][xIndex])./dx[:,iDimension][xIndex] )  )
            # calculate the index of nearest point
            
            # In linear interpolation; there is two weights per dimension
            #                                weight 1      weight 2
            weightsCurrentDimension = [1.0.-cellFraction cellFraction]
            
            # Calculate weights
            # After the for loop finishes; the rows of weight sum to 1 as a check.
            # multiply the weights from each dimension
            weight = weight.*weightsCurrentDimension[:, localCellIndex[iDimension,:]]
            
            # compute the index corresponding to the weight 先看二维的[0 0 1 1]  
            xWeightIndex[iDimension] .=  repeat(xIndex,1,4).+ (localCellIndex[iDimension,:].-1)'
            
        end
        
        # clean up a little 暂时不释放
        #clear(getname(cellFraction), getname(xIndex), getname(weightsCurrentDimension), getname(localCellIndex))
        
        # calculate linear index
        xWeightIndex = subscript2index(nGrid, xWeightIndex)
        
        # Form the sparse Afidelity matrix for fidelity equations
        Afidelity = sparse(vec(repeat((1:nScatteredPoints),1,2^nDimensions)), vec(xWeightIndex), vec(weight), nScatteredPoints, nTotalGridPoints)
        
    elseif interpMethod == "cubic"
        # This will be needed below.
        # Each cubic interpolation has 4^nDimension nodes. The local 
        # dimension index label is 1; 2; 3; | 4 for each dimension. For 
        # instance; cubic interpolation in 2d has 16 nodes with the 
        # following indexes:
        #    node label  =  1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16
        # localCellIndex = [1 1 1 1 2 2 2 2 3 3  3  3  4  4  4  4
        #                   1 2 3 4 1 2 3 4 1 2  3  4  1  2  3  4]
        if nDimensions == 2
            localCellIndex = [1 1 1 1 2 2 2 2 3 3 3 3 4 4 4 4
                              1 2 3 4 1 2 3 4 1 2 3 4 1 2 3 4]
        elseif nDimensions == 1
            localCellIndex = [1 2 3 4]
        end
        # Preallocate before loop
        weight = ones(nScatteredPoints, 4^nDimensions)
        xWeightIndex = ((zeros(Int,size(x,1),4)),)
        for iDimension = 1:nDimensions
            # Find cell index. Determine the cell the x-points lie in the
            # current xGrid dimension.
            h1 = StatsBase.fit(Histogram,x[:,iDimension],xGrid[iDimension],closed=:left)
            xIndex = StatsBase.binindex.(Ref(h1), x[:,iDimension]) 
            # Calculate low index used in cubic interpolation. 4 points are
            # needed  for cubic interpolation. The low index corresponds to
            # the smallest grid point used in the interpolation. The min()
            # & max ensures that the boundaries of the grid are
            # respected. For example; given a point x = 1.6 & a xGrid =
            # [0,1,2,3,4,5]. The points used for cubic interpolation would
            # be [0,1,2,3]. If x = 0.5, the points used would be [0,1,2,3]
            # this respects the bounds of the grid. If x = 4.9; the points
            # used would be [2,3,4,5]; again this respects the bounds of
            # the grid. 
            xIndex = min.(max.(xIndex .-1,1), nGrid[iDimension]-3)
            
            # Setup to calculate the 1d weights in the current dimension.
            # The 1d weights are based on cubic Lagrange polynomial
            # interpolation. The alphas & betas below help keep the
            # calculation readable & also save on a few floating point
            # operations at the cost of memory. There are 4 cubic Lagrange
            # polynomials that correspond to the weights. They have the
            # following form
            #
            # p1[x] = (x-x2)/(x1-x2)*(x-x3)/(x1-x3)*(x-x4)/(x1-x4) 
            # p2[x] = (x-x1)/(x2-x1)*(x-x3)/(x2-x3)*(x-x4)/(x2-x4) 
            # p3[x] = (x-x1)/(x3-x1)*(x-x2)/(x3-x2)*(x-x4)/(x3-x4) 
            # p4[x] = (x-x1)/(x4-x1)*(x-x2)/(x4-x2)*(x-x3)/(x4-x3)
            # 
            # The alphas & betas are defined as follows
            # alpha1 = x - x1
            # alpha2 = x - x2
            # alpha3 = x - x3
            # alpha4 = x - x4
            #
            # beta12 = x1 - x2
            # beta13 = x1 - x3
            # beta14 = x1 - x4
            # beta23 = x2 - x3
            # beta24 = x2 - x4
            # beta34 = x3 - x4
            alpha1 = x[:,iDimension] .- xGrid[iDimension][xIndex]
            alpha2 = x[:,iDimension] .- xGrid[iDimension][xIndex.+1]
            alpha3 = x[:,iDimension] .- xGrid[iDimension][xIndex.+2]
            alpha4 = x[:,iDimension] .- xGrid[iDimension][xIndex.+3]
            beta12 = xGrid[iDimension][xIndex] .- xGrid[iDimension][xIndex.+1]
            beta13 = xGrid[iDimension][xIndex] .- xGrid[iDimension][xIndex.+2]
            beta14 = xGrid[iDimension][xIndex] .- xGrid[iDimension][xIndex.+3]
            beta23 = xGrid[iDimension][xIndex.+1] .- xGrid[iDimension][xIndex.+2]
            beta24 = xGrid[iDimension][xIndex.+1] .- xGrid[iDimension][xIndex.+3]
            beta34 = xGrid[iDimension][xIndex.+2] .- xGrid[iDimension][xIndex.+3]
            
            weightsCurrentDimension = [ alpha2./beta12.*alpha3./beta13.*alpha4./beta14  -alpha1./beta12.*alpha3./beta23.*alpha4./beta24  alpha1./beta13.*alpha2./beta23.*alpha4./beta34  -alpha1./beta14.*alpha2./beta24.*alpha3./beta34]
    
            # Accumulate the weight contribution for each dimension by
            # multiplication. After the for loop finishes; the rows of
            # weight sum to 1 as a check
            weight = weight.*weightsCurrentDimension[:, collect(localCellIndex[iDimension,:]')][:,1,:]
            
            # compute the index corresponding to the weight
            xWeightIndex[iDimension] .=  repeat(xIndex,1,4) .+ (localCellIndex[iDimension,:].-1)'
        end
        
        # clean up a little 暂不清理
        #= clear(getname(alpha1), getname(alpha2), getname(alpha3), getname(alpha4), ...
              getname(beta12), getname(beta13), getname(beta14), getname(beta23), ...
              getname(beta24), getname(beta34), getname(weightsCurrentDimension), ...
              getname(xIndex), getname(localCellIndex)) =#
        
        # convert linear index
        xWeightIndex = subscript2index(nGrid, xWeightIndex)
         # Form the sparse Afidelity matrix for fidelity equations
        Afidelity = sparse(vec(repeat((1:nScatteredPoints),1,4^nDimensions)), vec(xWeightIndex), vec(weight), nScatteredPoints, nTotalGridPoints)
        
    else
        error("interpMethod error.")
    end
    
    # clean up
    #= clear(getname(dx), getname(weight), getname(x), getname(xWeightIndex)) =#
    
    
    ## Smoothness Equations_____________________________________________________________________________________
    
    ### calculate the number of smoothness equations in each dimension
    
    # nEquations is a square matrix where the ith row contains
    # number of smoothing equations in each dimension. For instance; if the
    # nGrid is [ 3 6 7 8] & ith row is 2, nEquationPerDimension contains
    # [3 4 7 8]. Therefore, the nSmoothnessEquations is 3*4*7*8=672 for 2nd dimension [2nd row].
    nEquationsPerDimension = repeat(nGrid, nDimensions,1)
    nEquationsPerDimension = nEquationsPerDimension - 2*I
    nSmoothnessEquations = prod(nEquationsPerDimension,dims=2)
    
    # Calculate the total number of Smooth equations
    nTotalSmoothnessEquations = sum(nSmoothnessEquations)
    
    ### Calculate regularization matrices
    
    
    # compute the index multiplier for each dimension. This is used for calculating the linear index.
    multiplier = cumprod(nGrid,dims=2)
    # Preallocate the regularization equations
    Lreg = ((spzeros(nSmoothnessEquations[1],multiplier[1])),)

    # loop over each dimension. calculate numerical 2nd derivatives weights.
    for iDimension=1:nDimensions
        #if smoothness[iDimension] .== 0 #先不考虑s=0!!!
    #=         nTotalSmoothnessEquations = nTotalSmoothnessEquations - nSmoothnessEquations[iDimension]
            Lreg[iDimension] = []
            
            # In the special case you try to fit a lookup table with no
            # smoothing; index1; index2; & index3 do not exist. The clear()
            # statement later would throw an error if index1; index2; &
            # index3 did not exist.
            index1=[]
            index2=[]
            index3=[] =#
        #else
            # initialize the index for the first grid vector
            if iDimension==1
                index1 = 1:nGrid[1]-2
                index2 = 2:nGrid[1]-1
                index3 = 3:nGrid[1]
            else
                index1 = 1:nGrid[1]
                index2 = index1
                index3 = index1
            end
            
            # loop over dimensions accumulating the contribution to the linear
            # index vector in each dimension. Note this section of code works very
            # similar to combining ndgrid & sub2ind. Basically; inspiration came
            # from looking at ndgrid & sub2ind.
#=             for iCell = 2:nDimensions  #只关于二维
                if iCell == iDimension   #二维中的第二维
                    index1 = vec(index1 .+ ((1:nGrid[iCell]-2)'.-1)*multiplier[iCell-1])
                    index2 = vec(index2 .+ ((2:nGrid[iCell]-1)'.-1)*multiplier[iCell-1])
                    index3 = vec(index3 .+ ((3:nGrid[iCell])'.-1)*multiplier[iCell-1])
                else    #二维中的第一维
                    currentDimensionIndex = (1:nGrid[iCell])'
                    index1 = vec(index1 .+ (currentDimensionIndex.-1)*multiplier[iCell-1])
                    index2 = vec(index2 .+ (currentDimensionIndex.-1)*multiplier[iCell-1])
                    index3 = vec(index3 .+ (currentDimensionIndex.-1)*multiplier[iCell-1])
                end
            end =#
            
            
            # Scales as if there is the same number of residuals along the
            # current dimension as there are fidelity equations total; use the
            # square root because the residuals will be squared to minimize
            # squared error.
            smoothnessScale = sqrt(nScatteredPoints/nSmoothnessEquations[iDimension])
            
            # Axis Scaling. This is equivalent to normalizing the current axis
            # to 0 to 1. i.e. If you scale one axis; the same smoothness factor
            # can be used to get similar shaped topology.
            axisScale = (xGridMax[iDimension] - xGridMin[iDimension]).^2
    
            
            # Create the Lreg for each dimension & store it a cell array.
            Lreg[iDimension] .= sparse(vec(repeat((1:nSmoothnessEquations[iDimension]),1,3)),vec([index1 index2 index3]),
                vec(smoothness[iDimension]*smoothnessScale*axisScale*secondDerivativeWeights(xGrid[iDimension],nGrid[iDimension],iDimension, nGrid)),
                  nSmoothnessEquations[iDimension], nTotalGridPoints)
        #end
    end
    
    # clean up & free up memory()
    #clear(getname(index1), getname(index2), getname(index3), getname(xGrid))
    
    ## Assemble & Solve the Overall Equation System
    
    # concatenate the fidelity equations & smoothing equations together
    A = [Afidelity; Lreg[1]]
    
    # clean up
    # clear(getname(Afidelity), getname(Lreg)); 
    
    # solve the full system()_______________________________________________________________________________________
    
    if solver == "anti division"
                yGrid = A \ collect(vec([y; spzeros(nTotalSmoothnessEquations,1)]))
    elseif solver == "normal"
                yGrid = (A'*A)\ collect(vec(A'*[y; spzeros(nTotalSmoothnessEquations,1)]))
    #elseif solver ==  "lsqr', 'pcg', 'symmlq"    
    #=         switch solver
                case ("pcg', 'symmlq")
                    # setup needed normal equation matrices
                    AA = A'*A
                    d = A"*[y spzeros(nTotalSmoothnessEquations,1)]"
                    
                    # clean up
                    clear(getname(A), getname(y))
                    
                    # calculate preconditioner if possible
                    [M, preconditioner] = calculatePreconditioner[AA]
                    
                    # Call pcg | symmlq differently depending on the preconditioner
                    switch preconditioner
                        case "none"
                            [yGrid, solverExitFlag] = feval(solver, AA, d, solverTolerance, maxIterations); ##ok<FVAL>
                        case "ichol"
                            [yGrid, solverExitFlag] = feval(solver, AA, d, solverTolerance, maxIterations, M, M'); ##ok<FVAL>
                        otherwise()
                            error("Code should never reach this. Something is wrong with the preconditioner switch statement. Fix it.")
                    end # end pcg; symmlq preconditioner switch statement
                    
                case "lsqr"
                    # calculate preconditioner if possible
                    [M, preconditioner] = calculatePreconditioner[A'*A]
                    
                    # Call lsqr differently depending on the preconditioner
                    switch preconditioner
                        case "none"
                            [yGrid, solverExitFlag] = lsqr(A,[y spzeros(nTotalSmoothnessEquations,1)]', solverTolerance, maxIterations)
                        case "ichol"
                            [yGrid, solverExitFlag] = lsqr(A,[y spzeros(nTotalSmoothnessEquations,1)]", solverTolerance, maxIterations, M")
                        otherwise()
                            error("Code should never reach this. Something is wrong with the preconditioner switch statement. Fix it.")
                    end # end lsqr preconditioner switch statement
                    
                otherwise()
                    error("Code should never reach this. Something is wrong with iterative solver switch statement.")
            end # end iterative solver switch block =#
            
            # Check the iterative solver flag
            #= switch solverExitFlag
                case 0
                    # Do nothing. This is good.
                case 1
                    warning("#s iterated #d times but did not converge.", solver, maxIterations)
                case 2
                    warning("The #s preconditioner was ill-conditioned.", solver)
                case 3
                    warning("#s stagnated. (Two consecutive iterates were the same.)", solver)
                case 4
                    warning("During #s solving, one of the scalar quantities calculated during pcg became too small | too large to continue computing.", solver)
                otherwise()
                    error("Code should never reach this. Something is wrong with iterative flag switch block.")
            end # iterative solver flag switch block 
            =#
    else    
        error("solver error.")
    end  # switch solver
    
    # convert to a full column vector
    yGrid = vec(yGrid) 
    
    # reshape if needed
    if nDimensions > 1
        yGrid = reshape(yGrid, nGrid[1], nGrid[2])
    end
    
    return yGrid
    
end #end function regularizeNd