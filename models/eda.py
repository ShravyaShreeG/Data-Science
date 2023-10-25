import pandas as pd
import matplotlib.pyplot as plt
from wordcloud import WordCloud
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.manifold import TSNE
import umap
import seaborn as sns
import regex as re
from sklearn.feature_extraction.text import CountVectorizer


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

dtm = df_dtm.drop('Product_title',axis=1)

# # Word frequency ------------------------------------------------------------------
# word_frequency = pd.DataFrame(dtm.sum())
# word_frequency.reset_index(inplace=True)
# word_frequency.columns = ['Word', 'Frequency']
#
# word_frequency = word_frequency.sort_values(by='Frequency', ascending=False)
# word_frequency= word_frequency.head(20)
# print(word_frequency)


# WORD CLOUD--------------------------------------------------------------------------------------------

def word_cloud(word_frequency):
    word_frequency_dict = dict(zip(word_frequency['Word'], word_frequency['Frequency']))
    # Create a WordCloud object with the desired parameters
    font_path = '/Users/tinydragon/scrapyThings/scrapyThings/07558_CenturyGothic.ttf'
    wordcloud = WordCloud(width=800, height=800, background_color='white',font_path=font_path).generate_from_frequencies(word_frequency_dict)
    # Display the generated word cloud using matplotlib
    plt.figure(figsize=(10, 5))
    plt.imshow(wordcloud, interpolation='bilinear')
    plt.axis('off')  # Remove axis ticks and labels
    plt.show()

# # VERTICAL BAR CHART -------------------------------------------------------------------------------------
#
# # Set the figure size
# plt.figure(figsize=(10, 6))
#
# # Create the bar chart
# colors = plt.cm.viridis(np.linspace(0, 1, len(word_frequency)))
# plt.barh(word_frequency['Word'], word_frequency['Frequency'], color= colors)
#
# # Add labels and title
# plt.xlabel('Frequency')
# plt.ylabel('Ingredients')
# plt.title('Top 20 Ingredients based on Frequency')
# plt.xticks(rotation=45)
# plt.tight_layout()
# plt.show()


# TOP 160 CORRELATED TERMS--------------------------------------------------------------------

correlation_matrix = dtm.corr()
# print(correlation_matrix)
top_correlated_terms = correlation_matrix.stack().sort_values(ascending=False)
df_cor_terms = top_correlated_terms.reset_index()
df_cor_terms.columns = ['Term1', 'Term2', 'Corr']
print(df_cor_terms.shape)

df_cor_terms = df_cor_terms[df_cor_terms['Term1'] != df_cor_terms['Term2']]
sorted_df = df_cor_terms.sort_values(by='Corr', ascending=False)
top_160_cor_terms_pairs= sorted_df[sorted_df['Corr']>0.9]
print(top_160_cor_terms_pairs.shape)
shuffled_df = top_160_cor_terms_pairs.sample(frac=1, random_state=42).head(20)  # Use a random_state for reproducibility
print(shuffled_df)


# top_160_corr_terms = list(top_160_cor_terms_pairs['Term1']) + list(top_160_cor_terms_pairs['Term2'])
# top_160_corr_terms = list(dict.fromkeys(top_160_corr_terms))
# print(top_160_corr_terms)


# # BASIC STATISTICS
#
# # Calculate basic statistics for term frequencies
# mean_term_freqs = np.mean(dtm, axis=1)
# median_term_freqs = np.median(dtm, axis=1)
# std_term_freqs = np.std(dtm, axis=1)
# percentiles_term_freqs = np.percentile(dtm, [25, 50, 75], axis=1)
#
# # Print the calculated statistics for each document
# for document_index in range(dtm.shape[0]):
#     document_statistics = {
#         "Document": document_index,
#         "Mean": mean_term_freqs[document_index],
#         "Median": median_term_freqs[document_index],
#         "Standard Deviation": std_term_freqs[document_index],
#         "25th Percentile": percentiles_term_freqs[0, document_index],
#         "50th Percentile (Median)": percentiles_term_freqs[1, document_index],
#         "75th Percentile": percentiles_term_freqs[2, document_index]
#     }
#
#     # print("Statistics for Document", document_index)
#     # for stat, value in document_statistics.items():
#     #     print(f"{stat}: {value}")
#
# word_frequency = dtm.sum().values
# # print(word_frequency)
# max_freq = np.max(word_frequency)
# min_freq = np.min(word_frequency)
# mean_term_freqs = np.mean(word_frequency)
# median_term_freqs = np.median(word_frequency)
# std_term_freqs = np.std(word_frequency)
# percentiles_term_freqs = np.percentile(word_frequency, [25, 50, 75])
# #
# # print(f'mean_term_freqs:{mean_term_freqs}')
# # print(f'median_term_freqs:{median_term_freqs}')
# # print(f'std_term_freqs:{std_term_freqs}')
# # print(f'percentiles_term_freqs:{percentiles_term_freqs}')
# # print(f'max_freq:{max_freq}')
# # print(f'min_freq:{min_freq}')
#
# # # Plot a box plot for word frequencies
# # plt.figure(figsize=(10, 6))
# # plt.boxplot(word_frequency)
# # plt.xlabel('Words')
# # plt.ylabel('Word Frequency')
# # plt.title('Box Plot of Word Frequencies')
# # plt.xticks([])
# # plt.show()
#
#
# # # Other basics
# # print(f'Shape of Document term matrix: {dtm.shape}')
# #
# # # Calculate the sparsity ratio
# # sparsity_ratio = 1.0 - (np.count_nonzero(dtm) / dtm.size)
# # sparsity_percentage = 100.0 * (1.0 - (np.count_nonzero(dtm) / dtm.size))
# # print(f'sparsity ratio: {sparsity_ratio}')
# # print(f'sparcity percentage: {sparsity_percentage}')
# #
#
# # # JACCARD SIMILARITY AND DOCUMENT CLUSTERS
# #
# # # Calculate cosine similarity matrix
# # cosine_sim_matrix = cosine_similarity(dtm)
# #
# # # Perform dimensionality reduction using UMAP
# # reducer = umap.UMAP(n_components=2, random_state=42)
# # embedding = reducer.fit_transform(cosine_sim_matrix)
# # print(embedding)
# #
# # # # Assuming you have 'cluster_labels' assigned from a clustering algorithm
# # # cluster_labels = [0, 1, 0, 1, 2, 2, 0, 2, 1]  # Replace with your cluster labels
# # #
# # # # Create a scatter plot to visualize clusters using seaborn
# # # plt.figure(figsize=(10, 8))
# # # sns.scatterplot(x=embedding[:, 0], y=embedding[:, 1], hue=cluster_labels, palette='Set1')
# # # plt.title('Document Clusters Visualization')
# # # plt.xlabel('UMAP Dimension 1')
# # # plt.ylabel('UMAP Dimension 2')
# # # plt.legend(title='Cluster')
# # # plt.show()
#

# # HARMFUL BOTANICAL EDA---------------------------------------------------------------------------
# df_om_grouped_source = df_om.groupby('Source')['harmful'].count().reset_index()
# df = df_om[df_om['harmful']==1]
#
# # for index, row in df.iterrows():
# #     print(row['Matched'])
#
# grouped_df = df.groupby('Source')['harmful'].sum().reset_index()
#
# # Find the ratio of harmful products found per website
# grouped_df['Total_count'] = df_om_grouped_source['harmful']
# grouped_df['Ratio'] = grouped_df['harmful']/grouped_df['Total_count']
# sorted_df = grouped_df.sort_values(by='Ratio', ascending=False)
# print(sorted_df)
#
# df['match_string'] = df['Matched'].apply(lambda x: re.sub(r"[\[\]']", '', x))
# df['match_string'] = df['match_string'].apply(lambda x: re.sub(r',\s*', ',', x))
# # print(df)
#
# documents = df['match_string'].tolist()
# # clear leading and trailing spaces
# documents = df['match_string'].apply(lambda x: str(x).strip()).tolist()
#
# vectorizer = CountVectorizer(tokenizer=lambda x: x.split(','))
# matrix = vectorizer.fit_transform(documents)
#
# dtm = pd.DataFrame(matrix.toarray(), columns=vectorizer.get_feature_names_out())
# # print(f'dtm.shape: {dtm.shape}')
# # print(f'harmful ingredients matched in products: {dtm.keys()}')


# # Word frequency ------------------------------------------------------------------
# word_frequency = pd.DataFrame(dtm.sum())
# # print(word_frequency)
# word_frequency.reset_index(inplace=True)
# word_frequency.columns = ['Word', 'Frequency']
# word_frequency = word_frequency.sort_values(by='Frequency', ascending=False)
# print(word_frequency)
#
# word_cloud(word_frequency)
