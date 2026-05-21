!=======================================================================
! 主程序：SFM-CNN 岩相智能预测系统
! 功能：协调地震正演、CNN特征提取与MPS随机模拟
!=======================================================================
program SFM_CNN_Facies_Prediction
    use data_manager
    use cnn_module
    use mps_engine
    use sfm_module
    implicit none
    
    ! 全局参数
    integer :: nx, ny, nz, nfacies, nrealizations
    integer :: ti_nx, ti_ny, ti_nz
    integer :: cnn_nfilters, cnn_ksize
    real    :: dt, dx, dy, dz
    character(len=256) :: parfile, ti_file, seismic_file, output_prefix
    
    ! 指针变量（动态分配）
    real, pointer :: TI(:,:,:)           ! 训练图像
    real, pointer :: SIM(:,:,:)          ! 模拟实现
    real, pointer :: SEIS(:,:,:)         ! 地震数据
    real, pointer :: CNN_FILTERS(:,:,:,:)! CNN卷积核
    real, pointer :: CNN_BIAS(:)         ! CNN偏置
    real, pointer :: ROTMAT(:,:,:)       ! 旋转矩阵（各向异性）
    real, pointer :: FEATURE_MAP(:,:,:,:)! CNN特征图
    
    integer :: ireal, i, j, k
    real :: start_time, end_time
    
    call cpu_time(start_time)
    
    ! 步骤1：读取参数
    call getarg(1, parfile)
    if(len_trim(parfile) == 0) parfile = 'sfm_cnn.par'
    call read_parameters(parfile, nx, ny, nz, nfacies, nrealizations, &
                         ti_nx, ti_ny, ti_nz, cnn_nfilters, cnn_ksize, &
                         dx, dy, dz, dt, ti_file, seismic_file, output_prefix)
    
    print *, '===================================================='
    print *, 'SFM-CNN 岩相智能预测系统'
    print *, '===================================================='
    print *, '模拟网格:', nx, 'x', ny, 'x', nz
    print *, '实现数量:', nrealizations
    print *, 'CNN滤波器:', cnn_nfilters, '核尺寸:', cnn_ksize
    
    ! 步骤2：分配内存
    call allocate_data(TI, ti_nx, ti_ny, ti_nz, 'TI')
    call allocate_data(SEIS, nx, ny, nz, 'SEIS')
    call allocate_data(SIM, nx, ny, nz, 'SIM')
    call allocate_data(CNN_FILTERS, cnn_ksize, cnn_ksize, cnn_ksize, cnn_nfilters, 'FILTERS')
    call allocate_data(CNN_BIAS, cnn_nfilters, 'BIAS')
    call allocate_data(ROTMAT, 3, 3, nz, 'ROTMAT')
    call allocate_data(FEATURE_MAP, nx, ny, nz, cnn_nfilters, 'FEATURES')
    
    ! 步骤3：读取训练图像与地震数据
    call read_index_file(ti_file, TI, ti_nx, ti_ny, ti_nz)
    call read_index_file(seismic_file, SEIS, nx, ny, nz)
    
    ! 步骤4：初始化CNN（加载预训练权重或随机初始化）
    call initialize_cnn(CNN_FILTERS, CNN_BIAS, cnn_ksize, cnn_nfilters)
    
    ! 步骤5：设置各向异性旋转矩阵
    call set_rotation_matrix(ROTMAT, 3, 3, nz, 45.0, 30.0, 0.0)
    
    ! 步骤6：主模拟循环
    do ireal = 1, nrealizations
        print *, '正在生成实现:', ireal, '/', nrealizations
        
        ! 6.1 生成随机路径
        call random_path(nx, ny, nz, SIM)
        
        ! 6.2 CNN特征提取（对地震数据）
        call convolution_3d(SEIS, nx, ny, nz, CNN_FILTERS, CNN_BIAS, &
                           cnn_ksize, cnn_nfilters, FEATURE_MAP)
        
        ! 6.3 SFM约束下的MPS模拟
        call simulation_mps_sfm(TI, ti_nx, ti_ny, ti_nz, SIM, nx, ny, nz, &
                               SEIS, FEATURE_MAP, cnn_nfilters, ROTMAT, &
                               nfacies, ireal)
        
        ! 6.4 保存实现
        call save_realization(SIM, nx, ny, nz, output_prefix, ireal)
    end do
    
    ! 步骤7：释放内存
    call deallocate_data(TI, 'TI')
    call deallocate_data(SEIS, 'SEIS')
    call deallocate_data(SIM, 'SIM')
    call deallocate_data(CNN_FILTERS, 'FILTERS')
    call deallocate_data(CNN_BIAS, 'BIAS')
    call deallocate_data(ROTMAT, 'ROTMAT')
    call deallocate_data(FEATURE_MAP, 'FEATURES')
    
    call cpu_time(end_time)
    print *, '全部完成！耗时:', end_time - start_time, '秒'
    
end program SFM_CNN_Facies_Prediction
