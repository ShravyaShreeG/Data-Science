import pandas as pd
from sklearn.cluster import KMeans
from scipy.spatial.distance import hamming
from scipy.spatial.distance import cdist
from sklearn.metrics import silhouette_score
from scipy.spatial.distance import pdist, squareform
from scipy.cluster.hierarchy import linkage, fcluster
import matplotlib.pyplot as plt
import seaborn as sns


pd.set_option('display.max_columns', None)  # Show all columns
pd.set_option('display.expand_frame_repr', False)

df_om = pd.read_csv('/Users/tinydragon/scrapyThings/scrapyThings/matrices/output_matrix.csv')
# print(df_om.head())
# print(df_om.keys())

df_dtm = pd.read_csv('/Users/tinydragon/scrapyThings/scrapyThings/matrices/document_term_matrix.csv')
# print(df_dtm.head())
# print(df_dtm.keys())

df_dtm_rm = pd.read_csv('/Users/tinydragon/scrapyThings/scrapyThings/matrices/document_term_matrix_rm.csv')
# print(df_dtm.head())
# print(df_dtm.keys())

df_pca= pd.read_csv('/Users/tinydragon/scrapyThings/scrapyThings/matrices/pca_df.csv')

df_chi = pd.read_csv('/Users/tinydragon/scrapyThings/scrapyThings/matrices/chi_squared_160.csv')

X = df_pca.copy()

# # kmeans---------------------------------------------------------------------------------------------
#
# distance_matrix = cdist(X, X, metric='hamming')
# k_values = []
# silhouette_scores = []
#
# for k in range(2, 10):
#     kmeans_model = KMeans(n_clusters=k, random_state=42)
#     cluster_labels = kmeans_model.fit_predict(distance_matrix)
#     silhouette_avg = silhouette_score(X, cluster_labels)
#
#     k_values.append(k)
#     silhouette_scores.append(silhouette_avg)
#
# # # Plot the silhouette scores for different k values
# # plt.plot(k_values, silhouette_scores, marker='o')
# # plt.xlabel('Number of Clusters (k)')
# # plt.ylabel('Silhouette Score')
# # plt.title('Silhouette Score vs. Number of Clusters')
# # plt.grid(True)
# # plt.show()
#
# df = pd.DataFrame({'k_val': k_values, 'sil_val': silhouette_scores})
# print(df)
# max_s_index = df['sil_val'].idxmax()
# max_k_value = df.loc[max_s_index, 'k_val']
#
# print(f"The value of k for which s is maximized: {max_k_value}")
# print(f"The value of s is : {df['sil_val'].max()}")


# # Heirarchical clustering---------------------------------------------------------------------------
#
# # Calculate the Jaccard distance matrix using pdist
# jaccard_distance = pdist(X, metric='jaccard')
#
# # Convert the Jaccard distance vector to a square distance matrix
# jaccard_distance_matrix = squareform(jaccard_distance)
#
# # Perform hierarchical clustering using 'linkage' function with Jaccard distance
# Z = linkage(jaccard_distance_matrix, method='single')
#
# # Set the distance threshold to form clusters (Option 1: Distance Threshold)
# distance_threshold = 0.5  # Adjust this value as needed
# cluster_labels = fcluster(Z, t=distance_threshold, criterion='distance')
#
# # Alternatively, specify the number of clusters directly (Option 2: Number of Clusters)
# num_clusters = 160  # Adjust this value as needed
# cluster_labels = fcluster(Z, t=num_clusters, criterion='maxclust')
#
# # Print the cluster assignments for each document
# for i in range(len((X))):
#     print(f"Document {i+1}: Cluster {cluster_labels[i]}")
#
#
# print(len(set(cluster_labels)))

# silhouette_avg = silhouette_score(X, cluster_labels)
# print("Silhouette Score:", silhouette_avg)

# USING PCA FEATURES FOR CLUSTERING

X = df_pca.copy()
X = X.iloc[:, :2]

# # kmeans---------------------------------------------------------------------------------------------

k_values = []
silhouette_scores = []
wcss = []

for k in range(2, 11):
    kmeans_model = KMeans(n_clusters=k, random_state=42)
    cluster_labels = kmeans_model.fit_predict(X)
    silhouette_avg = silhouette_score(X, cluster_labels)

    k_values.append(k)
    silhouette_scores.append(silhouette_avg)
    wcss.append(kmeans_model.inertia_)

# Plot the elbow curve
plt.plot(range(2, 11), wcss, marker='o')
plt.title('Elbow Method')
plt.xlabel('Number of Clusters')
plt.ylabel('WCSS')
plt.show()

# Plot the silhouette scores for different k values
plt.plot(k_values, silhouette_scores, marker='o')
plt.xlabel('Number of Clusters (k)')
plt.ylabel('Silhouette Score')
plt.title('Silhouette Score vs. Number of Clusters')
plt.grid(True)
plt.show()

df = pd.DataFrame({'k_val': k_values, 'sil_val': silhouette_scores})
max_s_index = df['sil_val'].idxmax()
max_k_value = df.loc[max_s_index, 'k_val']

print(f"The value of k for which s is maximized: {max_k_value}")
print(f"The value of s is : {df['sil_val'].max()}")


# PLOT PAIRWISE SCATTER PLOTS

# Optimal clusters

kmeans_model = KMeans(n_clusters=3, random_state=42)
cluster_labels = kmeans_model.fit_predict(X)

X['cluster'] = cluster_labels

# Create pairwise scatter plots colored by clusters
sns.set(style="ticks")
sns.pairplot(X, hue="cluster", diag_kind='kde')
plt.show()

# Create pairwise scatter plots colored by clusters
sns.set(style="ticks")
combined_df = pd.concat([X.iloc[:, :-1], df_om['harmful']], axis=1)
harmful = combined_df[combined_df['harmful']==1]
sns.pairplot(harmful.iloc[:,:-1], diag_kind='kde',palette= "red")
plt.show()

# print(X.keys())
df_cluster = pd.DataFrame({'Cluster_label': X['cluster'], 'Harmful': df_om['harmful']})
# df_cluster_grouped = df_cluster[df_cluster['Harmful']== 1].groupby('Cluster_label')['Harmful'].sum().reset_index()

df_cluster_grouped = df_cluster.groupby('Cluster_label').agg({'Harmful': 'sum', 'Harmful': 'count'}).reset_index()
df_cluster_grouped.rename(columns={'Harmful': 'total'}, inplace=True)
print(df_cluster_grouped)

# # ratio of harmful products in the 3 clusters
# df_cluster_grouped['ratio']= df_cluster_grouped['Harmful']/df_cluster_grouped['Harmful'].sum()
#
# # ratio of harmful products to total products in 3 clusters
# df_cluster_grouped['ratio_to_total']= df_cluster_grouped['Harmful']/df_om['harmful'].count()
# print(df_cluster_grouped)
#
# df_om['Cluster_label']= X['cluster']
#
# print(df_om.keys())
#
# # df_om.to_csv('df_cluster.csv',index=False)
#
#

