import numpy as np
import pandas as pd
from sklearn.feature_selection import chi2
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.metrics import roc_curve, auc
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
import numpy as np
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

X= df_dtm_rm.drop('Product_title', axis=1)
# print(X.keys())

target = df_om['harmful']

# Perform the chi-squared test for feature selection
chi2_scores, p_values = chi2(X, target)
df_chi = pd.DataFrame({'Term': X.columns, 'Chi_score': chi2_scores, 'p-value': p_values})
df_chi = df_chi[df_chi['p-value'] < 0.001]
df_chi = df_chi.sort_values(by='Chi_score', ascending=False)
df_chi = df_chi[df_chi['Chi_score']>10]
df_chi = df_chi
print(df_chi.shape)
print(df_chi)

selected_terms = list(df_chi['Term'])
# print(selected_terms)

X = X[selected_terms]
# print(X)

# X.to_csv('chi_squared_160.csv', index=False)

# Split the data into training and testing sets
y = target  # Labels
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state= 123)

# # Fit the logistic regression model
# model = LogisticRegression()
# model.fit(X_train, y_train)
#
# # Make predictions on the testing set
# y_pred = model.predict(X_test)
#
# # Evaluate the model's performance
# accuracy = accuracy_score(y_test, y_pred)
# precision = precision_score(y_test, y_pred, zero_division=1)
# recall = recall_score(y_test, y_pred, zero_division=1)
# f1 = f1_score(y_test, y_pred, zero_division=1)
#
# print(f"Accuracy: {accuracy:.2f}")
# print(f"Precision: {precision:.2f}")
# print(f"Recall: {recall:.2f}")
# print(f"F1-score: {f1:.2f}")

def roc(X_test,y_test,model):

    y_prob = model.predict_proba(X_test)[:, 1]
    # Calculate the false positive rate (FPR), true positive rate (TPR), and threshold values for the ROC curve
    fpr, tpr, thresholds = roc_curve(y_test, y_prob)

    # Calculate the area under the ROC curve
    roc_auc = auc(fpr, tpr)
    print(f'AUC: {roc_auc:.2f}')

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


def plot_confusion_matrix(cm, classes, normalized=True, cmap='BuPu'):
    plt.figure(figsize=[10, 8])
    norm_cm = cm
    if normalized:
        norm_cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        sns.heatmap(norm_cm, annot=cm, fmt='g', xticklabels=classes, yticklabels=classes, cmap=cmap)
    plt.show()

# OVERSAMPLING-------------------------------------------------------------
from imblearn.over_sampling import RandomOverSampler
from sklearn.linear_model import LogisticRegression
from imblearn.over_sampling import SMOTE

# RANDOM OVERSAMPLING
oversampler = RandomOverSampler(sampling_strategy='auto', random_state=42)
# Fit and apply the oversampler to the training data
X_train_resampled, y_train_resampled = oversampler.fit_resample(X_train, y_train)

# Apply SMOTE to the training data
smote = SMOTE(sampling_strategy='auto', random_state=4)
X_train_resampled, y_train_resampled = smote.fit_resample(X_train, y_train)

# Initialize and train a classifier (e.g., Logistic Regression) on the resampled data
classifier = LogisticRegression()
classifier.fit(X_train_resampled, y_train_resampled)
y_pred = classifier.predict(X_test)

# Evaluate the model's performance
accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred, zero_division=1)
recall = recall_score(y_test, y_pred, zero_division=1)
f1 = f1_score(y_test, y_pred, zero_division=1)

print(f"Accuracy: {accuracy:.2f}")
print(f"Precision: {precision:.2f}")
print(f"Recall: {recall:.2f}")
print(f"F1-score: {f1:.2f}")

roc(X_test,y_test,classifier)
plot_confusion_matrix(confusion_matrix(y_test,y_pred), ['0', '1'])


# ADABOOST

def adaboost(X_train, X_test, y_train, y_test):

    from sklearn.ensemble import AdaBoostClassifier
    from sklearn.tree import DecisionTreeClassifier

    # Create a decision tree as the base estimator (weak classifier)
    base_estimator = DecisionTreeClassifier(max_depth=1)

    # Create the AdaBoost classifier
    adaboost_classifier = AdaBoostClassifier(base_estimator=base_estimator, n_estimators=50, random_state=42)

    # Fit the AdaBoost classifier to the training data
    adaboost_classifier.fit(X_train, y_train)

    # Make predictions on the test set
    y_pred = adaboost_classifier.predict(X_test)

    roc(X_test, y_test, adaboost_classifier)
    plot_confusion_matrix(confusion_matrix(y_test, y_pred), ['0', '1'])

    # Evaluate the model's performance
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred, zero_division=1)
    recall = recall_score(y_test, y_pred, zero_division=1)
    f1 = f1_score(y_test, y_pred, zero_division=1)

    print(f"Accuracy: {accuracy:.2f}")
    print(f"Precision: {precision:.2f}")
    print(f"Recall: {recall:.2f}")
    print(f"F1-score: {f1:.2f}")


# adaboost(X_train, X_test, y_train, y_test)
