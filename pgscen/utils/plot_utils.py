
import seaborn as sns
from scipy.spatial import distance
from scipy.cluster.hierarchy import linkage, dendrogram

cov_cmap = sns.diverging_palette(3, 237, s=81, l=43, sep=3, as_cmap=True)


def get_clustermat(cov_mat):
    clust_order = dendrogram(linkage(distance.pdist(cov_mat,
                                                    metric='euclidean'),
                                     method='centroid'),
                             no_plot=True)['leaves']

    return cov_mat.iloc[clust_order, clust_order]
