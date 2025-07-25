import numpy as np
import time


def calc_SED(pA, pI, pB):
    """
    计算同步欧氏距离（SED）误差
    :param pA: 起始点
    :param pI: 中间点
    :param pB: 终止点
    :return: SED误差
    """
    pA_lat, pA_lon, pA_time = pA
    pI_lat, pI_lon, pI_time = pI
    pB_lat, pB_lon, pB_time = pB

    middle_dist = pI_time - pA_time
    total_dist = pB_time - pA_time
    if total_dist == 0:
        time_ratio = 0
    else:
        time_ratio = middle_dist / total_dist

    # 线性插值计算理论位置
    lat = pA_lat + (pB_lat - pA_lat) * time_ratio
    lon = pA_lon + (pB_lon - pA_lon) * time_ratio

    lat_diff = lat - pI_lat
    lon_diff = lon - pI_lon
    error = np.sqrt((lat_diff * lat_diff) + (lon_diff * lon_diff))
    return error


def calc_DP(pA, pI, pB):
    """
    计算垂直距离（Perpendicular Distance, PD）
    :param pA: 起始点
    :param pI: 中间点
    :param pB: 终止点
    :return: 最短距离
    """
    pA_lat, pA_lon, pA_time = pA
    pI_lat, pI_lon, pI_time = pI
    pB_lat, pB_lon, pB_time = pB

    # 直线方程参数
    A = pA_lon - pB_lon
    B = - (pA_lat - pB_lat)
    C = pA_lat * pB_lon - pB_lat * pA_lon

    if A == 0 and B == 0:
        shortDist = 0
    else:
        shortDist = abs((A * pI_lat + B * pI_lon + C) / np.sqrt(A * A + B * B))

    return shortDist


def calc_AVS(pA, pI, pB):
    """
    计算速度绝对值差（AVS）
    :param pA: 起始点
    :param pI: 中间点
    :param pB: 终止点
    :return: AVS值
    """
    pA_lat, pA_lon, pA_time = pA
    pI_lat, pI_lon, pI_time = pI
    pB_lat, pB_lon, pB_time = pB

    d1 = np.sqrt((pI_lat - pA_lat) * (pI_lat - pA_lat) + (pI_lon - pA_lon) * (pI_lon - pA_lon))
    d2 = np.sqrt((pB_lat - pI_lat) * (pB_lat - pI_lat) + (pB_lon - pI_lon) * (pB_lon - pI_lon))

    v1 = 0
    v2 = 0
    if (pI_time - pA_time) > 0:
        v1 = d1 / (pI_time - pA_time)
    if (pB_time - pI_time) > 0:
        v2 = d2 / (pB_time - pI_time)
    AVS = abs(v2 - v1)

    return AVS


def calc_TR_SP(trajectory, dim_set, traj_time, epsilon, epsilon2, calc_func, calc_func2):
    """
    使用两种压缩技术递归压缩轨迹
    :param trajectory: 单条轨迹或其片段
    :param dim_set: 轨迹属性集合
    :param traj_time: 每个点的时间（秒）
    :param epsilon: 第一种压缩的阈值
    :param epsilon2: 第二种压缩的阈值
    :param calc_func: 第一种压缩的距离度量
    :param calc_func2: 第二种压缩的距离度量
    :return: 压缩后的轨迹（dict）
    """
    new_trajectory = {}
    for dim in dim_set:
        new_trajectory[dim] = np.array([])
    traj_len = len(trajectory['lat'])

    # 计算最大距离及其索引
    dmax, idx, _ = traj_max_dists(trajectory, traj_time, calc_func)
    start_location = (trajectory['lat'][0], trajectory['lon'][0], traj_time[0])
    final_location = (trajectory['lat'][-1], trajectory['lon'][-1], traj_time[-1])
    middle_location = (trajectory['lat'][idx], trajectory['lon'][idx], traj_time[idx])
    d_idx = calc_func2(start_location, middle_location, final_location)

    trajectory['time'] = trajectory['time'].astype(str)

    # 若两种距离均大于阈值，则递归分段压缩
    if (dmax > epsilon) & (d_idx > epsilon2):
        traj1 = {}
        traj2 = {}
        for dim in dim_set:
            traj1[dim] = trajectory[dim][0:idx]
            traj2[dim] = trajectory[dim][idx:]

        # 递归压缩两段
        recResults1 = traj1
        if len(traj1['lat']) > 2:
            recResults1 = traj_compression(traj1, dim_set, traj_time[0:idx], calc_func, epsilon)

        recResults2 = traj2
        if len(traj2['lat']) > 2:
            recResults2 = traj_compression(traj2, dim_set, traj_time[idx:], calc_func, epsilon)

        for dim in dim_set:
            new_trajectory[dim] = np.append(new_trajectory[dim], recResults1[dim])
            new_trajectory[dim] = np.append(new_trajectory[dim], recResults2[dim])

    else:
        # 若不满足压缩条件，则保留首尾点
        trajectory['time'] = trajectory['time'].astype(str)
        for dim in dim_set:
            new_trajectory[dim] = np.append(new_trajectory[dim], trajectory[dim][0])
            if traj_len > 1:
                new_trajectory[dim] = np.append(new_trajectory[dim], trajectory[dim][-1])

    return new_trajectory


def traj_max_dists(trajectory, traj_time, calc_func):
    """
    计算所有中间点到首尾点的最大距离
    :param trajectory: 单条轨迹（dict）
    :param traj_time: 每个点的时间（秒）
    :param calc_func: 距离度量函数
    :return: 最大距离、最大距离的索引、平均距离
    """
    dmax = 0
    idx = 0
    ds = np.array([])
    traj_len = len(trajectory['lat'])
    # 起点和终点
    start_location = (trajectory['lat'][0], trajectory['lon'][0], traj_time[0])
    final_location = (trajectory['lat'][-1], trajectory['lon'][-1], traj_time[-1])
    for i in range(1, (traj_len - 1)):
        # 计算每个中间点的距离
        middle = (trajectory['lat'][i], trajectory['lon'][i], traj_time[i])
        d = calc_func(start_location, middle, final_location)
        ds = np.append(ds, d)
        if d > dmax:
            dmax = d
            idx = i

    return dmax, idx, ds.mean()


def traj_compression(trajectory, dim_set, traj_time, calc_func, epsilon):
    """
    使用指定压缩方法递归压缩轨迹
    :param trajectory: 单条轨迹或其片段
    :param dim_set: 轨迹属性集合
    :param traj_time: 每个点的时间（秒）
    :param calc_func: 距离度量函数
    :param epsilon: 阈值
    :return: 压缩后的轨迹（dict）
    """
    new_trajectory = {}
    for dim in dim_set:
        new_trajectory[dim] = np.array([])
    traj_len = len(trajectory['lat'])

    # 计算最大距离及其索引
    dmax, idx, _ = traj_max_dists(trajectory, traj_time, calc_func)
    trajectory['time'] = trajectory['time'].astype(str)

    # 若最大距离大于阈值，递归分段压缩
    if dmax > epsilon:
        traj1 = {}
        traj2 = {}
        for dim in dim_set:
            traj1[dim] = trajectory[dim][0:idx]
            traj2[dim] = trajectory[dim][idx:]

        recResults1 = traj1
        if len(traj1['lat']) > 2:
            recResults1 = traj_compression(traj1, dim_set, traj_time[0:idx], calc_func, epsilon)

        recResults2 = traj2
        if len(traj2['lat']) > 2:
            recResults2 = traj_compression(traj2, dim_set, traj_time[idx:], calc_func, epsilon)

        for dim in dim_set:
            new_trajectory[dim] = np.append(new_trajectory[dim], recResults1[dim])
            new_trajectory[dim] = np.append(new_trajectory[dim], recResults2[dim])

    else:
        # 不满足压缩条件，保留首尾点
        trajectory['time'] = trajectory['time'].astype(str)
        for dim in dim_set:
            new_trajectory[dim] = np.append(new_trajectory[dim], trajectory[dim][0])
            if traj_len > 1:
                new_trajectory[dim] = np.append(new_trajectory[dim], trajectory[dim][-1])

    return new_trajectory


def compression(dataset, metric='TR', verbose=True, alpha=1):
    """
    压缩整个轨迹数据集
    :param dataset: 轨迹数据集（dict）
    :param metric: 压缩方法或组合
    :param verbose: 是否打印信息
    :param alpha: 压缩因子
    :return: 压缩后数据集、压缩率、处理时间
    """
    # sys.setrecursionlimit(2200)
    metrics = {'TR': calc_SED,
               'DP': calc_DP,
               'SP': calc_AVS,
               'TR_SP': calc_TR_SP,
               'SP_TR': calc_TR_SP,
               'SP_DP': calc_TR_SP,
               'DP_SP': calc_TR_SP,
               'DP_TR': calc_TR_SP,
               'TR_DP': calc_TR_SP}

    calc_func = metrics[metric]

    mmsis = list(dataset.keys())
    new_dataset = {}
    compression_rate = np.array([])
    processing_time = np.array([])

    dim_set = dataset[mmsis[0]].keys()

    if verbose:
        print(f"Compressing with {metric} and factor {alpha}")
    for id_mmsi in range(len(mmsis)):
        new_dataset[mmsis[id_mmsi]] = {}
        if verbose:
            print(f"\tCompressing {id_mmsi} of {len(mmsis)}")
        # 当前轨迹
        t0 = time.time()
        curr_traj = dataset[mmsis[id_mmsi]]
        # 获取时间（归一化秒）
        traj_time = curr_traj['time'].astype('datetime64[s]')
        traj_time = np.hstack((0, np.diff(traj_time).cumsum().astype('float')))
        traj_time = traj_time / traj_time.max()
        # 压缩轨迹
        compress_traj = curr_traj
        try:
            if metric in ['TR_SP']:
                max_epsilon, idx, epsilon = traj_max_dists(curr_traj, traj_time, calc_SED)
                max_epsilon2, idx2, epsilon2 = traj_max_dists(curr_traj, traj_time, calc_AVS)
                compress_traj = calc_func(curr_traj, dim_set, traj_time, epsilon * alpha, epsilon2 * alpha, calc_SED,
                                          calc_AVS)
            elif metric in ['SP_TR']:
                max_epsilon, idx, epsilon = traj_max_dists(curr_traj, traj_time, calc_AVS)
                max_epsilon2, idx2, epsilon2 = traj_max_dists(curr_traj, traj_time, calc_SED)
                compress_traj = calc_func(curr_traj, dim_set, traj_time, epsilon * alpha, epsilon2 * alpha, calc_AVS,
                                          calc_SED)
            elif metric in ['SP_DP']:
                max_epsilon, idx, epsilon = traj_max_dists(curr_traj, traj_time, calc_AVS)
                max_epsilon2, idx2, epsilon2 = traj_max_dists(curr_traj, traj_time, calc_DP)
                compress_traj = calc_func(curr_traj, dim_set, traj_time, epsilon * alpha, epsilon2 * alpha, calc_AVS,
                                          calc_DP)
            elif metric in ['TR_DP']:
                max_epsilon, idx, epsilon = traj_max_dists(curr_traj, traj_time, calc_SED)
                max_epsilon2, idx2, epsilon2 = traj_max_dists(curr_traj, traj_time, calc_DP)
                compress_traj = calc_func(curr_traj, dim_set, traj_time, epsilon * alpha, epsilon2 * alpha, calc_SED,
                                          calc_DP)
            elif metric in ['DP_SP']:
                max_epsilon, idx, epsilon = traj_max_dists(curr_traj, traj_time, calc_DP)
                max_epsilon2, idx2, epsilon2 = traj_max_dists(curr_traj, traj_time, calc_AVS)
                compress_traj = calc_func(curr_traj, dim_set, traj_time, epsilon * alpha, epsilon2 * alpha, calc_DP,
                                          calc_AVS)
            elif metric in ['DP_TR']:
                max_epsilon, idx, epsilon = traj_max_dists(curr_traj, traj_time, calc_DP)
                max_epsilon2, idx2, epsilon2 = traj_max_dists(curr_traj, traj_time, calc_SED)
                compress_traj = calc_func(curr_traj, dim_set, traj_time, epsilon * alpha, epsilon2 * alpha, calc_DP,
                                          calc_SED)
            else:
                max_epsilon, idx, epsilon = traj_max_dists(curr_traj, traj_time, calc_func)
                compress_traj = traj_compression(curr_traj, dim_set, traj_time, calc_func, epsilon * alpha)
        except:
            print(
                f"\t\tIt was not possible to compress this trajectory {mmsis[id_mmsi]} of length {len(curr_traj['lat'])}.")

        compress_traj['time'] = compress_traj['time'].astype('datetime64[s]')
        new_dataset[mmsis[id_mmsi]] = compress_traj
        t1 = time.time() - t0
        # 计算压缩率和耗时
        compression_rate = np.append(compression_rate, 1 - (len(compress_traj['lat']) / len(curr_traj['lat'])))
        processing_time = np.append(processing_time, t1)

    return new_dataset, compression_rate, processing_time
