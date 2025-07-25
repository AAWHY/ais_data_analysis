import numpy as np
import math

from typing import Callable, Dict
from numba import jit, types, int32, int64
from numba import typed
from timeit import default_timer as timer


class DiscreteFrechet(object):
    """
    计算两条折线（轨迹）之间的离散Fréchet距离，使用递归算法
    """

    def __init__(self, dist_func):
        """
        初始化，设置点对距离函数
        :param dist_func: 距离函数，接受两个点的坐标
        """
        self.dist_func = dist_func
        self.ca = np.array([0.0])

    def distance(self, p: np.ndarray, q: np.ndarray) -> float:
        """
        计算两条折线p和q的Fréchet距离
        :param p: 折线p
        :param q: 折线q
        :return: Fréchet距离
        """
        def calculate(i: int, j: int) -> float:
            """
            递归计算p[i]和q[j]之间的距离
            """
            if self.ca[i, j] > -1.0:
                return self.ca[i, j]

            d = self.dist_func(p[i], q[j])
            if i == 0 and j == 0:
                self.ca[i, j] = d
            elif i > 0 and j == 0:
                self.ca[i, j] = max(calculate(i-1, 0), d)
            elif i == 0 and j > 0:
                self.ca[i, j] = max(calculate(0, j-1), d)
            elif i > 0 and j > 0:
                self.ca[i, j] = max(min(calculate(i-1, j),
                                        calculate(i-1, j-1),
                                        calculate(i, j-1)), d)
            else:
                self.ca[i, j] = np.infty
            return self.ca[i, j]

        n_p = p.shape[0]
        n_q = q.shape[0]
        self.ca = np.zeros((n_p, n_q))
        self.ca.fill(-1.0)
        return calculate(n_p - 1, n_q - 1)

@jit(nopython=True)
def _get_linear_frechet(p: np.ndarray, q: np.ndarray,
                        dist_func: Callable[[np.ndarray, np.ndarray], float]) -> np.ndarray:
    """
    线性方式计算Fréchet距离的辅助函数
    """
    n_p = p.shape[0]
    n_q = q.shape[0]
    ca = np.zeros((n_p, n_q), dtype=np.float64)

    for i in range(n_p):
        for j in range(n_q):
            d = dist_func(p[i], q[j])

            if i > 0 and j > 0:
                ca[i, j] = max(min(ca[i - 1, j],
                                   ca[i - 1, j - 1],
                                   ca[i, j - 1]), d)
            elif i > 0 and j == 0:
                ca[i, j] = max(ca[i - 1, 0], d)
            elif i == 0 and j > 0:
                ca[i, j] = max(ca[0, j - 1], d)
            elif i == 0 and j == 0:
                ca[i, j] = d
            else:
                ca[i, j] = np.infty
    return ca

class LinearDiscreteFrechet(DiscreteFrechet):
    """
    线性离散Fréchet距离计算类
    """
    def __init__(self, dist_func):
        DiscreteFrechet.__init__(self, dist_func)
        # JIT编译
        self.distance(np.array([[0.0, 0.0], [1.0, 1.0]]),
                      np.array([[0.0, 0.0], [1.0, 1.0]]))

    def distance(self, p: np.ndarray, q: np.ndarray) -> float:
        n_p = p.shape[0]
        n_q = q.shape[0]
        self.ca = _get_linear_frechet(p, q, self.dist_func)
        return self.ca[n_p - 1, n_q - 1]

@jit(nopython=True)
def distance_matrix(p: np.ndarray,
                    q: np.ndarray,
                    dist_func: Callable[[np.array, np.array], float]) -> np.ndarray:
    """
    计算两条轨迹所有点对之间的距离矩阵
    """
    n_p = p.shape[0]
    n_q = q.shape[0]
    dist = np.zeros((n_p, n_q), dtype=np.float64)
    for i in range(n_p):
        for j in range(n_q):
            dist[i, j] = dist_func(p[i], q[j])
    return dist

@jit(nopython=True)
def _bresenham_pairs(x0: int, y0: int, x1: int, y1: int) -> np.ndarray:
    """
    生成对角线坐标（Bresenham算法）
    """
    dx = abs(x1 - x0)
    dy = abs(y1 - y0)
    dim = max(dx, dy)
    pairs = np.zeros((dim, 2), dtype=np.int64)
    x, y = x0, y0
    sx = -1 if x0 > x1 else 1
    sy = -1 if y0 > y1 else 1
    if dx > dy:
        err = dx // 2
        for i in range(dx):
            pairs[i, 0] = x
            pairs[i, 1] = y
            err -= dy
            if err < 0:
                y += sy
                err += dx
            x += sx
    else:
        err = dy // 2
        for i in range(dy):
            pairs[i, 0] = x
            pairs[i, 1] = y
            err -= dx
            if err < 0:
                x += sx
                err += dy
            y += sy
    return pairs

@jit(int64(int32, int32), nopython=True)
def rc(row: types.int32, col: types.int32) -> types.int64:
    """
    行列号合并为唯一key
    """
    return (row << 32) + col

@jit(nopython=True)
def _get_rc(a: Dict, row: types.int64, col: types.int64, d: types.float64 = np.inf) -> types.float64:
    """
    获取稀疏矩阵中的值
    """
    kk = rc(row, col)
    if kk in a:
        return a.get(kk)
    else:
        return d

@jit(nopython=True)
def _get_corner_min_sparse(f_mat: Dict, i: int, j: int) -> float:
    """
    稀疏矩阵取三邻域最小值
    """
    if i > 0 and j > 0:
        a = min(_get_rc(f_mat, i - 1, j - 1),
                _get_rc(f_mat, i, j - 1),
                _get_rc(f_mat, i - 1, j))
    elif i == 0 and j == 0:
        a = f_mat.get(rc(i, j))
    elif i == 0:
        a = f_mat.get(rc(i, j - 1))
    else:  # j == 0:
        a = f_mat.get(rc(i - 1, j))
    return a

@jit(nopython=True)
def _get_corner_min_array(f_mat: np.ndarray, i: int, j: int) -> float:
    """
    普通数组取三邻域最小值
    """
    if i > 0 and j > 0:
        a = min(f_mat[i - 1, j - 1],
                f_mat[i, j - 1],
                f_mat[i - 1, j])
    elif i == 0 and j == 0:
        a = f_mat[i, j]
    elif i == 0:
        a = f_mat[i, j - 1]
    else:  # j == 0:
        a = f_mat[i - 1, j]
    return a

@jit(nopython=True)
def _fast_distance_sparse(p: np.ndarray, q: np.ndarray, diag: np.ndarray,
                         dist_func: Callable[[np.array, np.array], float]) -> Dict:
    """
    稀疏方式快速计算距离
    """
    n_diag = diag.shape[0]
    diag_max = 0.0
    i_min = 0
    j_min = 0
    p_count = p.shape[0]
    q_count = q.shape[0]

    # 创建稀疏距离字典
    dist = typed.Dict.empty(key_type=types.int64, value_type=types.float64)

    # 填充对角线
    for k in range(n_diag):
        i0 = diag[k, 0]
        j0 = diag[k, 1]
        d = dist_func(p[i0], q[j0])
        if d > diag_max:
            diag_max = d
        dist[rc(i0, j0)] = d

    for k in range(n_diag - 1):
        i0 = diag[k, 0]
        j0 = diag[k, 1]

        p_i0 = p[i0]
        q_j0 = q[j0]

        for i in range(i0 + 1, p_count):
            key = rc(i, j0)
            if key not in dist:
                d = dist_func(p[i], q_j0)
                if d < diag_max or i < i_min:
                    dist[key] = d
                else:
                    break
            else:
                break
        i_min = i

        for j in range(j0 + 1, q_count):
            key = rc(i0, j)
            if key not in dist:
                d = dist_func(p_i0, q[j])
                if d < diag_max or j < j_min:
                    dist[key] = d
                else:
                    break
            else:
                break
        j_min = j
    return dist

@jit(nopython=True)
def _fast_distance_matrix(p, q, diag, dist_func):
    """
    普通数组方式快速计算距离
    """
    n_diag = diag.shape[0]
    diag_max = 0.0
    i_min = 0
    j_min = 0
    p_count = p.shape[0]
    q_count = q.shape[0]

    # 创建距离矩阵
    dist = np.full((p_count, q_count), np.inf, dtype=np.float64)

    # 填充对角线
    for k in range(n_diag):
        i0 = diag[k, 0]
        j0 = diag[k, 1]
        d = dist_func(p[i0], q[j0])
        diag_max = max(diag_max, d)
        dist[i0, j0] = d

    for k in range(n_diag - 1):
        i0 = diag[k, 0]
        j0 = diag[k, 1]
        p_i0 = p[i0]
        q_j0 = q[j0]

        for i in range(i0 + 1, p_count):
            if np.isinf(dist[i, j0]):
                d = dist_func(p[i], q_j0)
                if d < diag_max or i < i_min:
                    dist[i, j0] = d
                else:
                    break
            else:
                break
        i_min = i

        for j in range(j0 + 1, q_count):
            if np.isinf(dist[i0, j]):
                d = dist_func(p_i0, q[j])
                if d < diag_max or j < j_min:
                    dist[i0, j] = d
                else:
                    break
            else:
                break
        j_min = j
    return dist

@jit(nopython=True)
def _fast_frechet_sparse(dist, diag: np.ndarray, p: np.ndarray, q: np.ndarray):
    """
    稀疏方式快速计算Frechet距离
    """
    for k in range(diag.shape[0]):
        i0 = diag[k, 0]
        j0 = diag[k, 1]

        for i in range(i0, p.shape[0]):
            key = rc(i, j0)
            if key in dist:
                c = _get_corner_min_sparse(dist, i, j0)
                if c > dist[key]:
                    dist[key] = c
            else:
                break

        # Add 1 to j0 to avoid recalculating the diagonal
        for j in range(j0 + 1, q.shape[0]):
            key = rc(i0, j)
            if key in dist:
                c = _get_corner_min_sparse(dist, i0, j)
                if c > dist[key]:
                    dist[key] = c
            else:
                break
    return dist

@jit(nopython=True)
def _fast_frechet_matrix(dist: np.ndarray, diag: np.ndarray, p: np.ndarray, q: np.ndarray) -> np.ndarray:
    """
    普通数组方式快速计算Frechet距离
    """
    for k in range(diag.shape[0]):
        i0 = diag[k, 0]
        j0 = diag[k, 1]

        for i in range(i0, p.shape[0]):
            if np.isfinite(dist[i, j0]):
                c = _get_corner_min_array(dist, i, j0)
                if c > dist[i, j0]:
                    dist[i, j0] = c
            else:
                break

        # Add 1 to j0 to avoid recalculating the diagonal
        for j in range(j0 + 1, q.shape[0]):
            if np.isfinite(dist[i0, j]):
                c = _get_corner_min_array(dist, i0, j)
                if c > dist[i0, j]:
                    dist[i0, j] = c
            else:
                break
    return dist

@jit(nopython=True)
def _fdfd_sparse(p: np.ndarray, q: np.ndarray, dist_func: Callable[[np.array, np.array], float]) -> float:
    """
    稀疏方式快速计算Frechet距离主入口
    """
    diagonal = _bresenham_pairs(0, 0, p.shape[0], q.shape[0])
    ca = _fast_distance_sparse(p, q, diagonal, dist_func)
    ca = _fast_frechet_sparse(ca, diagonal, p, q)
    return ca

@jit(nopython=True)
def _fdfd_matrix(p: np.ndarray, q: np.ndarray, dist_func: Callable[[np.array, np.array], float]) -> float:
    """
    普通数组方式快速计算Frechet距离主入口
    """
    diagonal = _bresenham_pairs(0, 0, p.shape[0], q.shape[0])
    ca = _fast_distance_matrix(p, q, diagonal, dist_func)
    ca = _fast_frechet_matrix(ca, diagonal, p, q)
    return ca

class FastDiscreteFrechetSparse(object):
    """
    稀疏加速离散Frechet距离计算类
    """
    def __init__(self, dist_func):
        """
        初始化
        """
        self.times = []
        self.dist_func = dist_func
        self.ca = typed.Dict.empty(key_type=types.int64,
                                   value_type=types.float64)
        # JIT编译
        self.distance(np.array([[0.0, 0.0], [1.0, 1.0]]),
                      np.array([[0.0, 0.0], [1.0, 1.0]]))

    def timed_distance(self, p: np.ndarray, q: np.ndarray) -> float:
        """
        计时计算Frechet距离
        """
        start = timer()
        diagonal = _bresenham_pairs(0, 0, p.shape[0], q.shape[0])
        self.times.append(timer() - start)

        start = timer()
        ca = _fast_distance_sparse(p, q, diagonal, self.dist_func)
        self.times.append(timer() - start)

        start = timer()
        ca = _fast_frechet_sparse(ca, diagonal, p, q)
        self.times.append(timer() - start)

        self.ca = ca
        return ca[rc(p.shape[0]-1, q.shape[0]-1)]

    def distance(self, p: np.ndarray, q: np.ndarray) -> float:
        """
        计算Frechet距离
        """
        ca = _fdfd_sparse(p, q, self.dist_func)
        self.ca = ca
        return ca[rc(p.shape[0]-1, q.shape[0]-1)]

@jit(nopython=True, fastmath=True)
def haversine(p: np.ndarray, q: np.ndarray) -> float:
    """
    向量化haversine距离计算
    :p: 起点（弧度）
    :q: 终点（弧度）
    :return: 球面距离
    """
    d = q - p
    a = math.sin(d[0]/2.0)**2 + math.cos(p[0]) * math.cos(q[0]) * math.sin(d[1]/2.0)**2
    c = 2 * math.atan2(math.sqrt(a), math.sqrt(1.0 - a))
    return c

def fast_frechet(p, q):
    """
    计算两条轨迹的离散Frechet距离（haversine度量）
    """
    fast_frechet = LinearDiscreteFrechet(haversine)
    distance = fast_frechet.distance(p, q)
    return distance
