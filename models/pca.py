import pandas as pd
from sklearn.decomposition import TruncatedSVD
import numpy as np
from sklearn.decomposition import SparsePCA
import matplotlib.pyplot as plt
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score


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


X= df_dtm_rm.drop('Product_title', axis=1)
print(X.keys())


# PCA----------------------------------------------------------------------------------------

# Step 1: Create the PCA object
num_components = 20  # You can choose the number of components you want to keep
pca = PCA(n_components=num_components)

# Step 2: Fit the PCA model to the data
pca.fit(X)

# Step 3: Transform the data into the new PCA space
transformed_data = pca.transform(X)
pc_columns = ['PC{}'.format(i + 1) for i in range(num_components)]
transformed_df = pd.DataFrame(transformed_data, columns= pc_columns)
print("Original Data Shape:", X.shape)
print("Transformed Data Shape:", transformed_data.shape)
print("Transformed Data:")
print(transformed_df)

X = transformed_df

y = df_om['harmful']  # Labels
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=123)

# Fit the logistic regression model
model = LogisticRegression()
model.fit(X_train, y_train)

# Make predictions on the testing set
y_pred = model.predict(X_test)

# Evaluate the model's performance
accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred, zero_division=1)
recall = recall_score(y_test, y_pred, zero_division=1)
f1 = f1_score(y_test, y_pred, zero_division=1)

print(f"Accuracy: {accuracy:.2f}")
print(f"Precision: {precision:.2f}")
print(f"Recall: {recall:.2f}")
print(f"F1-score: {f1:.2f}")

# Check explained variance ratio
explained_variance_ratio = pca.explained_variance_ratio_
print("Explained Variance Ratio:")
print(explained_variance_ratio)

# Check cumulative explained variance
cumulative_explained_variance = pca.explained_variance_ratio_.cumsum()
print("Cumulative Explained Variance:")
print(cumulative_explained_variance)

# Plot the scree plot
plt.figure(figsize=(8, 5))
plt.plot(range(1, len(explained_variance_ratio) + 1), explained_variance_ratio, marker='o')
plt.title('Scree Plot')
plt.xlabel('Principal Component')
plt.ylabel('Explained Variance Ratio')
plt.xticks(np.arange(1, len(explained_variance_ratio) + 1))
plt.show()
# transformed_df.to_csv('pca_df.csv',index=False)

# BINARY PCA-----------------------------------------------------------------------------------------------

# df = pd.DataFrame(X)
#
# # Step 1: Convert to centered binary data
# centered_data = df - df.mean()
#
# # Step 2: Calculate the correlation matrix
# correlation_matrix = centered_data.corr()
#
# # Step 3: Perform Eigenvalue Decomposition
# eigenvalues, eigenvectors = np.linalg.eig(correlation_matrix)
#
# # Get real-valued eigenvectors and eigenvalues
# real_eigenvalues = np.real(eigenvalues)
# real_eigenvectors = np.real(eigenvectors)
#
# # Step 4: Select Principal Components (e.g., top-2 components)
# k = 9
#
# selected_components = real_eigenvectors[:, :k]
#
# # Step 5: Project Data
# reduced_data = centered_data.dot(selected_components)
#
# print(reduced_data)
#
# # Calculate explained variance for each principal component
# explained_variance = real_eigenvalues / np.sum(real_eigenvalues)
# k_explained_variance = explained_variance[:k]
#
# print("Explained Variance of Each Principal Component:")
# for i, variance in enumerate(k_explained_variance):
#     print(f"Principal Component {i+1}: {variance:.2f}")
#
# cumulative_explained_variance = np.cumsum(k_explained_variance)
# print(cumulative_explained_variance)
#
# print("Cumulative Explained Variance:")
# for i, variance in enumerate(cumulative_explained_variance):
#     print(f"Principal Component {i+1}: {variance:.2f}")
#

# # SPARSE PCA--------------------------------------------------------------------------------------------
#
#
# # Apply Sparse PCA
# n_components = 2  # Number of components to extract
# sparse_pca = SparsePCA(n_components=n_components)
# X_sparse_pca = sparse_pca.fit_transform(X)
#
# # Convert the result to a Pandas DataFrame for visualization and analysis
# df_sparse_pca = pd.DataFrame(X_sparse_pca, columns=[f"Component_{i+1}" for i in range(n_components)])
#
# print(df_sparse_pca)
# #

# # TRUNCATED SVD

# # Apply Truncated SVD
# n_components = 20  # Number of components to keep
# svd = TruncatedSVD(n_components=n_components)
# X_svd = svd.fit_transform(X)
#
# # Convert the result to a Pandas DataFrame for visualization and analysis
# df_svd = pd.DataFrame(X_svd, columns=[f"Component_{i+1}" for i in range(n_components)])
# print(df_svd)
#
# explained_variance = svd.explained_variance_ratio_
# cumulative_variance = np.cumsum(explained_variance)
#
# print("Explained Variance for each component:")
# for i in range(n_components):
#     print(f"Component_{i+1}: {explained_variance[i]}")
#
# print("\nCumulative Variance for each component:")
# for i in range(n_components):
#     print(f"Component_{i+1}: {cumulative_variance[i]}")
