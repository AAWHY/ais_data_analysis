import os
import numpy as np
import pandas as pd
import pickle
from src.distances import compute_distance_matrix
from src.clustering import Clustering
from sklearn import metrics
import mantel
from preprocessing.compress_trajectories import compress_trips, get_raw_dataset


def get_time(path):
    """
    读取距离计算的耗时矩阵，并返回其上三角部分的总耗时。
    :param path: 存储耗时矩阵的文件路径
    :return: 上三角的耗时数组
    """
    up = pickle.load(open(path, 'rb'))
    up = pd.DataFrame.from_dict(up)
    up[up.isna()] = 0
    up = up.to_numpy()
    up = up[np.triu_indices_from(up)]
    return up


def purity_score(y_true, y_pred):
    """
    计算聚类的纯度分数。
    :param y_true: 真实标签
    :param y_pred: 预测标签
    :return: purity分数
    """
    contingency_matrix = metrics.cluster.contingency_matrix(y_true, y_pred)
    return np.sum(np.amax(contingency_matrix, axis=0)) / np.sum(contingency_matrix)


def factor_analysis(dataset_path, compress_opt, folder):
    """
    针对不同压缩因子，评估压缩算法的压缩率和耗时，并保存结果。
    :param dataset_path: 数据集路径
    :param compress_opt: 压缩算法名称
    :param folder: 结果保存文件夹
    :return: 各因子的压缩率和耗时
    """
    factors = [2, 1.5, 1, 1/2, 1/4, 1/8, 1/16, 1/32, 1/64, 1/128]
    rates = pd.DataFrame()
    times = pd.DataFrame()
    for i in factors:
        # 对每个压缩因子进行轨迹压缩
        comp_dataset, comp_rate, comp_times = compress_trips(dataset_path, compress=compress_opt, alpha=i)
        rates = pd.concat([rates, pd.DataFrame(comp_rate)], axis=1)
        times = pd.concat([times, pd.DataFrame(comp_times)], axis=1)
    rates.columns = [str(i) for i in factors]
    times.columns = [str(i) for i in factors]
    # 保存压缩率和耗时到csv
    rates.to_csv(f'{folder}/{compress_opt}-compression_rates.csv', index=False)
    times.to_csv(f'{folder}/{compress_opt}-compression_times.csv', index=False)
    return rates, times


def factor_dist_analysis(dataset_path, compress_opt, folder, ncores=15, metric='dtw'):
    """
    针对不同压缩因子，评估压缩后轨迹的距离矩阵，并与原始距离矩阵做Mantel检验。
    :param dataset_path: 数据集路径
    :param compress_opt: 压缩算法名称
    :param folder: 结果保存文件夹
    :param ncores: 并行计算核数
    :param metric: 距离度量方法
    :return: mantel相关性等指标
    """
    factors = [2, 1.5, 1, 1/2, 1/4, 1/8, 1/16, 1/32, 1/64, 1/128]
    times = pd.DataFrame()
    measures = {}
    # 计算原始轨迹的距离矩阵
    features_folder = f'{folder}/NO/'
    if not os.path.exists(features_folder):
        os.makedirs(features_folder)
    features_path, main_time = compute_distance_matrix(get_raw_dataset(dataset_path), features_folder, verbose=True,
                                                       njobs=ncores, metric=metric)
    dist_raw = pickle.load(open(features_path, 'rb'))
    # 归一化处理
    dist_raw = dist_raw/dist_raw.max().max()
    dist_raw[np.isinf(dist_raw)] = dist_raw[~np.isinf(dist_raw)].max() + 1
    dist_raw[dist_raw < 0] = 0
    dist_raw = dist_raw/dist_raw.max().max()
    dist_raw_time = get_time(main_time)
    times = pd.concat([times, pd.DataFrame(dist_raw_time)], axis=1)
    for i in factors:
        # 针对每个压缩因子，计算压缩后轨迹的距离矩阵
        comp_dataset, comp_rate, comp_times = compress_trips(dataset_path, compress=compress_opt, alpha=i)
        features_folder = f'{folder}/{compress_opt}-{i}/'
        if not os.path.exists(features_folder):
            os.makedirs(features_folder)
        features_path, feature_time = compute_distance_matrix(comp_dataset, features_folder, verbose=True,
                                                                   njobs=ncores, metric=metric)
        dtw_factor = pickle.load(open(features_path, 'rb'))
        measures[i] = {}
        dtw_factor[np.isinf(dtw_factor)] = dtw_factor[~np.isinf(dtw_factor)].max() + 1
        dtw_factor[dtw_factor < 0] = 0
        dtw_factor = dtw_factor/dtw_factor.max().max()
        # Mantel检验，衡量压缩前后距离矩阵的相关性
        measures[i]['mantel-corr'], measures[i]['mantel-pvalue'], _ = mantel.test(dist_raw, dtw_factor,
                                                                                  method='pearson', tail='upper')
        dtw_factor_time = get_time(feature_time)
        times = pd.concat([times, pd.DataFrame(dtw_factor_time)], axis=1)
    measures = pd.DataFrame(measures)
    measures.columns = [str(i) for i in factors]
    measures.to_csv(f'{folder}/measures_{metric}_{compress_opt}_times.csv')
    times.columns = ['no'] + [str(i) for i in factors]
    times.to_csv(f'{folder}/{metric}_{compress_opt}_times.csv', index=False)
    return measures


def factor_cluster_analysis(dataset_path, compress_opt, folder, ncores=15, metric='dtw', mcs=2):
    """
    针对不同压缩因子，评估压缩后轨迹的聚类效果（NMI、MH等），并保存结果。
    :param dataset_path: 数据集路径
    :param compress_opt: 压缩算法名称
    :param folder: 结果保存文件夹
    :param ncores: 并行计算核数
    :param metric: 距离度量方法
    :param mcs: 聚类类别数
    :return: NMI分数
    """
    factors = [2, 1.5, 1, 1/2, 1/4, 1/8, 1/16, 1/32, 1/64, 1/128]
    measures_mh = {}
    measures_nmi = {}
    times_cl = {}
    # 计算原始轨迹的距离矩阵
    features_folder = f'{folder}/NO/'
    if not os.path.exists(features_folder):
        os.makedirs(features_folder)
    features_path, _ = compute_distance_matrix(get_raw_dataset(dataset_path), features_folder, verbose=True,
                                               njobs=ncores, metric=metric)
    # 原始轨迹聚类
    model = Clustering(ais_data_path=dataset_path, distance_matrix_path=features_path, folder=features_folder,
                       minClusterSize=mcs, norm_dist=True)
    times_cl['no'] = model.time_elapsed
    labels_raw = model.labels
    for i in factors:
        # 针对每个压缩因子，计算压缩后轨迹的距离矩阵并聚类
        comp_dataset, comp_rate, comp_times = compress_trips(dataset_path, compress=compress_opt, alpha=i)
        features_folder = f'{folder}/{compress_opt}-{i}/'
        if not os.path.exists(features_folder):
            os.makedirs(features_folder)
        features_path, feature_time = compute_distance_matrix(comp_dataset, features_folder, verbose=True,
                                                                   njobs=ncores, metric=metric)
        model = Clustering(ais_data_path=dataset_path, distance_matrix_path=features_path, folder=features_folder,
                           minClusterSize=mcs, norm_dist=True)
        times_cl[str(i)] = model.time_elapsed
        labels_factor = model.labels
        # 计算聚类纯度、覆盖率、MH分数和NMI分数
        measures_purity = purity_score(labels_raw, labels_factor)
        measures_coverage = purity_score(labels_factor, labels_raw)
        measures_mh[str(i)] = 2/(1/measures_purity + 1/measures_coverage)
        measures_nmi[str(i)] = metrics.normalized_mutual_info_score(labels_raw, labels_factor)
    # 保存结果
    measures_mh = pd.Series(measures_mh)
    measures_mh.to_csv(f'{folder}/clustering_{compress_opt}_mh.csv')
    measures_nmi = pd.Series(measures_nmi)
    measures_nmi.to_csv(f'{folder}/clustering_{compress_opt}_nmi.csv')
    times_cl = pd.Series(times_cl)
    times_cl.columns = ['no'] + [str(i) for i in factors]
    times_cl.to_csv(f'{folder}/clustering_{compress_opt}_times.csv')
    return measures_nmi

