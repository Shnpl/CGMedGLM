# encoding: utf-8
# # utils for clustering during data preprocessing
from tqdm import tqdm
class DisjointSet:
    def __init__(self, n):
        self.parent = list(range(n))

    def find(self, a):
        if self.parent[a] != a:
            self.parent[a] = self.find(self.parent[a])
        return self.parent[a]

    def union(self, a, b):
        root_a = self.find(a)
        root_b = self.find(b)
        if root_a != root_b:
            self.parent[root_b] = root_a
def get_connected_components(reachable_matrix):
    num_nodes = len(reachable_matrix)
    visited = [False] * num_nodes
    components = []

    def dfs(node, component):
        visited[node] = True
        component.append(node)
        for neighbor in range(num_nodes):
            if not visited[neighbor] and reachable_matrix[node][neighbor]:
                dfs(neighbor, component)

    for i in tqdm(range(num_nodes)):
        if not visited[i]:
            component = []
            dfs(i, component)
            components.append(component)

    return components
def block_diagonalize(matrix, threshold):
    rows = len(matrix)
    cols = len(matrix[0])
    visited = [[False] * cols for _ in range(rows)]
    blocks = []

    def dfs(row, col, block):
        if row < 0 or row >= rows or col < 0 or col >= cols or visited[row][col] or matrix[row][col] <= threshold:
            return
        visited[row][col] = True
        block.append((row, col))
        dfs(row - 1, col, block)  # 上方
        dfs(row + 1, col, block)  # 下方
        dfs(row, col - 1, block)  # 左侧
        dfs(row, col + 1, block)  # 右侧

    for i in range(rows):
        for j in range(cols):
            if not visited[i][j] and matrix[i][j] > threshold:
                block = []
                dfs(i, j, block)
                blocks.append(block)

    return blocks
def threshold_clustering(similarity_matrix, threshold):
    num_objects = len(similarity_matrix)
    visited = [False] * num_objects
    clusters = []

    def dfs(node, cluster):
        visited[node] = True
        cluster.append(node)
        for neighbor in range(num_objects):
            if not visited[neighbor] and similarity_matrix[node][neighbor] >= threshold:
                dfs(neighbor, cluster)

    for i in range(num_objects):
        if not visited[i]:
            cluster = []
            dfs(i, cluster)
            clusters.append(cluster)

    return clusters
import concurrent.futures

def threshold_clustering_parallel(similarity_matrix, threshold):
    num_objects = len(similarity_matrix)
    visited = [False] * num_objects
    clusters = []

    def dfs(node, cluster):
        visited[node] = True
        cluster.append(node)
        for neighbor in range(num_objects):
            if not visited[neighbor] and similarity_matrix[node][neighbor] >= threshold:
                dfs(neighbor, cluster)

    with concurrent.futures.ThreadPoolExecutor() as executor:
        futures = []
        for i in range(num_objects):
            if not visited[i]:
                cluster = []
                futures.append(executor.submit(dfs, i, cluster))
                clusters.append(cluster)

        # Wait for all DFS operations to complete
        concurrent.futures.wait(futures)

    return clusters
from sklearn.cluster import SpectralClustering

def spectral_clustering(similarity_matrix, num_clusters):
    # 创建谱聚类模型
    model = SpectralClustering(n_clusters=num_clusters, affinity='precomputed')
    
    # 执行聚类
    labels = model.fit_predict(similarity_matrix)
    
    return labels

def similarity2accessibility(matrix,thres):
    bool_matrix = matrix>thres
    last_bool_matrix = bool_matrix.copy()
    i = 0
    while(True):
        bool_matrix = bool_matrix | (bool_matrix @ last_bool_matrix)
        if (bool_matrix == last_bool_matrix).all():
            print(f"Converged at {i} iterations")
            break
        last_bool_matrix = bool_matrix.copy()
        print(f"Iteration {i} completed")
    return bool_matrix
    