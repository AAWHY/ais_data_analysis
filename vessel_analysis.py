import src.analysis as analysis
import src.plot_images as pli
import os

# 渔船数据路径
data_path = './data/crop/DCAIS_[30, 1001, 1002]_region_[37.6, 39, -122.9, -122.2]_01-04_to_30-06_trips.csv'
# 油轮数据路径（如需分析油轮，取消下行注释）
# data_path = './data/crop/DCAIS_[80, 81, 82, 83, 84, 85, 86, 87, 88, 89, 1017, 1024]_region_[47.5, 49.3, -125.5, -122.5]_01-04_to_30-06_trips.csv'

# 距离度量方法，可选：'dtw', 'md', 'dfd', 'hd'
metric = 'md'

# 聚类类别数：渔船为2，油轮为3
mcs = 2

# 结果文件夹路径设置
file_name = os.path.basename(data_path)
file_name = os.path.splitext(file_name)[0]
folder = f'./results/crop/{file_name}/{metric}/'
if not os.path.exists(folder):
    os.makedirs(folder)

print(folder)

# 压缩分析，不同算法的压缩率和耗时
rates_dp, times_dp = analysis.factor_analysis(data_path, 'DP', folder)
rates_tr, times_tr = analysis.factor_analysis(data_path, 'TR', folder)
rates_sp, times_sp = analysis.factor_analysis(data_path, 'SP', folder)
rates_tr_sp, times_tr_sp = analysis.factor_analysis(data_path, 'TR_SP', folder)
rates5_sp_tr, times_sp_tr = analysis.factor_analysis(data_path, 'SP_TR', folder)
rates6_sp_dp, times_sp_dp = analysis.factor_analysis(data_path, 'SP_DP', folder)
rates_dp_sp, times_dp_sp = analysis.factor_analysis(data_path, 'DP_SP', folder)
rates_tr_dp, times_tr_dp = analysis.factor_analysis(data_path, 'TR_DP', folder)
rates_dp_tr, times_dp_tr = analysis.factor_analysis(data_path, 'DP_TR', folder)

# 距离矩阵分析
measure_dp = analysis.factor_dist_analysis(data_path, 'DP', folder, metric=metric)
measure_rt = analysis.factor_dist_analysis(data_path, 'TR', folder, metric=metric)
measure_sp = analysis.factor_dist_analysis(data_path, 'SP', folder, metric=metric)
measure_tr_sp = analysis.factor_dist_analysis(data_path, 'TR_SP', folder, metric=metric)
measure_sp_tr = analysis.factor_dist_analysis(data_path, 'SP_TR', folder, metric=metric)
measure_dp_sp = analysis.factor_dist_analysis(data_path, 'DP_SP', folder, metric=metric)
measure_sp_dp = analysis.factor_dist_analysis(data_path, 'SP_DP', folder, metric=metric)
measure_tr_dp = analysis.factor_dist_analysis(data_path, 'TR_DP', folder, metric=metric)
measure_dp_tr = analysis.factor_dist_analysis(data_path, 'DP_TR', folder, metric=metric)


# 聚类分析
measure_nmi = analysis.factor_cluster_analysis(data_path, 'DP', folder, metric=metric, mcs=mcs)
measure_nmi_tr = analysis.factor_cluster_analysis(data_path, 'TR', folder, metric=metric, mcs=mcs)
measure_nmi_sp = analysis.factor_cluster_analysis(data_path, 'SP', folder, metric=metric, mcs=mcs)
measure_nmi_tr_sp = analysis.factor_cluster_analysis(data_path, 'TR_SP', folder, metric=metric, mcs=mcs)
measure_nmi_sp_tr = analysis.factor_cluster_analysis(data_path, 'SP_TR', folder, metric=metric, mcs=mcs)
measure_nmi_dp_sp = analysis.factor_cluster_analysis(data_path, 'DP_SP', folder, metric=metric, mcs=mcs)
measure_nmi_sp_dp = analysis.factor_cluster_analysis(data_path, 'SP_DP', folder, metric=metric, mcs=mcs)
measure_nmi_tr_dp = analysis.factor_cluster_analysis(data_path, 'TR_DP', folder, metric=metric, mcs=mcs)
measure_nmi_dp_tr = analysis.factor_cluster_analysis(data_path, 'DP_TR', folder, metric=metric, mcs=mcs)

pli.lines_compression(folder, metric=metric)

