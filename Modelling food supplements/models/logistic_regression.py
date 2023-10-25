import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.model_selection import learning_curve
from sklearn.metrics import roc_curve, auc
import matplotlib.pyplot as plt


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

df_cluster = pd.read_csv('/Users/tinydragon/scrapyThings/scrapyThings/matrices/df_cluster.csv')


df = df_dtm_rm
df['Harmful'] = df_om['harmful']
print(df.keys())

# Normal fitting with test and train set, dtm without harmful products-------------------------------------------------

# Split the data into training and testing sets
X = df.drop(['Harmful','Product_title'], axis=1)  # Features (DTM)
y = df['Harmful']  # Labels
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state= 123)

# Fit the logistic regression model
model = LogisticRegression()
model.fit(X_train, y_train)

# Make predictions on the testing set
y_pred = model.predict(X_test)

dtm = df_dtm_rm.drop(['Product_title','Harmful'], axis=1)


# # DTM with top 160 ingredients based on frequency------------------------------------------------------
#
# word_frequency = dtm.sum()
# sorted_terms = word_frequency.sort_values(ascending=False)
# top_n_words = sorted_terms.head(160)
# # for index, text in top_n_words.iteritems():
# #     print(index)
# #     print(text)
# dtm_filtered = df_dtm_rm[top_n_words.index].copy()
# print(dtm_filtered.keys())
# dtm_filtered.loc[:, 'Harmful'] = df_dtm_rm['Harmful']
# print(dtm_filtered.keys())
#
# # Fit the model with this new data frame
# df = dtm_filtered.copy()
#
# # Split the data into training and testing sets
# X = df.drop(['Harmful'], axis=1)  # Features (DTM)
# y = df['Harmful']  # Labels
# X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=123)

# # Fit the logistic regression model
# model = LogisticRegression()
# model.fit(X_train, y_train)
#
# # Make predictions on the testing set
# y_pred = model.predict(X_test)




# # TOP 160 Correlated terms, correlation with the output-------------------------------------------------------------------------------
#
# dtm_harmful = df_dtm_rm.drop(['Product_title'], axis=1)
# correlation_values = []
#
# # Iterate over each column (except 'classifier') and calculate the correlation with 'classifier'
# for column in dtm_harmful.columns:
#     if column != 'Harmful':
#         correlation = dtm_harmful[column].corr(dtm_harmful['Harmful'])
#         correlation_values.append((column, correlation))
#
# correlation_values.sort(key=lambda x: abs(x[1]), reverse=True)
# top_160_terms = correlation_values[:160]
# top_160_terms_list = [t[0] for t in top_160_terms]
# # print(top_160_terms_list)
#
# # Print the correlation values
# for col, corr in top_160_terms:
#     print(f"Correlation between {col} and classifier: {corr}")
#
# dtm_filtered = dtm[top_160_terms_list]
# print(dtm_filtered.shape)
#
# dtm_filtered.loc[:, 'Harmful'] = df_dtm_rm['Harmful']
# print(dtm_filtered.keys())
#
# # Fit the model with this new data frame
# df = dtm_filtered.copy()
#
# # Split the data into training and testing sets
# X = df.drop(['Harmful'], axis=1)  # Features (DTM)
# y = df['Harmful']  # Labels
# X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=123)

# # Fit the logistic regression model
# model = LogisticRegression()
# model.fit(X_train, y_train)
#
# # Make predictions on the testing set
# y_pred = model.predict(X_test)
#




# # Logistic Regression With Cluster Labels And Sources
#
# df = df_cluster.copy()
# print(df.keys())
# X = df[['Source', 'Cluster_label']]
# X = pd.get_dummies(X, columns=['Cluster_label'], prefix='Cluster')
# X = pd.get_dummies(X, columns=['Source'], prefix= None)
# print(X.columns)
# # Split the data into training and testing sets
# y = df['harmful']  # Labels
# X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=123)
#
# # # Fit the logistic regression model
# # model = LogisticRegression()
# # model.fit(X_train, y_train)
# #
# # # Make predictions on the testing set
# # y_pred = model.predict(X_test)



def roc(X_test,y_test, model):

    y_prob = model.predict_proba(X_test)[:, 1]
    fpr, tpr, thresholds = roc_curve(y_test, y_prob)
    roc_auc = auc(fpr, tpr)

    # Plot the ROC curve
    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, color='blue', lw=2, label='ROC curve (AUC = %0.2f)' % roc_auc)
    plt.plot([0, 1], [0, 1], color='gray', lw=2, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate (FPR)')
    plt.ylabel('True Positive Rate (TPR)')
    plt.title('ROC Curve')
    plt.legend(loc='lower right')
    plt.show()


# CONFUSION MATRIX--------------------------------------------------------------------\
from sklearn.metrics import confusion_matrix
import numpy as np
import seaborn as sns


def plot_confusion_matrix(cm, classes, normalized=True, cmap='BuPu'):
    plt.figure(figsize=[2, 2])
    norm_cm = cm
    if normalized:
        norm_cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        sns.heatmap(norm_cm, annot=cm, fmt='g', xticklabels=classes, yticklabels=classes, cmap=cmap, cbar = False)
    plt.show()


# # OVERSAMPLING-------------------------------------------------------------
# from imblearn.over_sampling import RandomOverSampler
# from sklearn.linear_model import LogisticRegression
#
#
# oversampler = RandomOverSampler(sampling_strategy='auto', random_state=123)
# # Fit and apply the oversampler to the training data
# X_train_resampled, y_train_resampled = oversampler.fit_resample(X_train, y_train)
#
# # Initialize and train a classifier (e.g., Logistic Regression) on the resampled data
# model = LogisticRegression()
# model.fit(X_train_resampled, y_train_resampled)
# y_pred = model.predict(X_test)


def learning_curves(model, X_train, y_train):

    # Create learning curve data
    train_sizes, train_scores, val_scores = learning_curve(model, X_train, y_train, train_sizes=np.linspace(0.1, 1.0, 10), cv=5)

    # Calculate mean and standard deviation for training and validation scores
    train_scores_mean = np.mean(train_scores, axis=1)
    train_scores_std = np.std(train_scores, axis=1)
    val_scores_mean = np.mean(val_scores, axis=1)
    val_scores_std = np.std(val_scores, axis=1)

    # Plot learning curves
    plt.figure(figsize=(10, 6))
    plt.title("Learning Curves")
    plt.xlabel("Training Examples")
    plt.ylabel("Score")
    plt.grid()
    plt.fill_between(
        train_sizes,
        train_scores_mean - train_scores_std,
        train_scores_mean + train_scores_std,
        alpha=0.1,
        color="r"
    )

    plt.fill_between(
        train_sizes,
        val_scores_mean - val_scores_std,
        val_scores_mean + val_scores_std,
        alpha=0.1,
        color="g"
    )
    plt.plot(train_sizes, train_scores_mean, "o-", color="r", label="Training Score")
    plt.plot(train_sizes, val_scores_mean, "o-", color="g", label="Validation Score")
    plt.legend(loc="best")
    plt.show()



# Evaluate the model's performance-----------------------------------------------------------------

accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred, zero_division=1)
recall = recall_score(y_test, y_pred, zero_division=1)
f1 = f1_score(y_test, y_pred, zero_division=1)

print(f"Accuracy: {accuracy:.2f}")
print(f"Precision: {precision:.2f}")
print(f"Recall: {recall:.2f}")
print(f"F1-score: {f1:.2f}")

roc(X_test,y_test,model)
plot_confusion_matrix(confusion_matrix(y_test,y_pred), ['0', '1'])
learning_curves(model, X_train, y_train)