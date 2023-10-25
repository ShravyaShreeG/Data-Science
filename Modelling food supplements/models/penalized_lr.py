from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
import pandas as pd
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.model_selection import cross_val_score
import numpy as np
from sklearn.metrics import roc_curve, auc
import matplotlib.pyplot as plt
from imblearn.over_sampling import RandomOverSampler

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

df = df_dtm_rm
df['Harmful'] = df_om['harmful']
# print(df.keys())

# Normal fitting with test and train set, dtm without harmful products-------------------------------------------------

# Split the data into training and testing sets
X = df.drop(['Harmful','Product_title'], axis=1)  # Features (DTM)
y = df['Harmful']  # Labels

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


# OVERSAMPLING

oversampler = RandomOverSampler(sampling_strategy='auto', random_state=42)
X_train_resampled, y_train_resampled = oversampler.fit_resample(X_train, y_train)

# Create a Logistic Regression model with L1 regularization (Lasso)
lasso_model = LogisticRegression(penalty='l1', solver='liblinear')

# List of C values to test
c_values = [0.001, 0.01, 0.1, 1, 10, 100, 1000]
# c_values = np.arange(0.0005, 0.005, 0.0001)

# Perform cross-validation for each C value
best_c = None
max_recall = -1  # Initialize with a very low value

for c in c_values:
    lasso_model.C = c  # Set the current C value
    scores = cross_val_score(lasso_model, X_train_resampled, y_train_resampled, cv=3, scoring='recall')
    print(f"C = {c}, Mean Recall: {scores.mean()}")

    mean_recall = scores.mean()
    print(f"c = {c}, Mean Recall: {mean_recall}")
    if mean_recall > max_recall:
        max_recall = mean_recall
        best_c = c

print(best_c)
# Train the final model with the best C value
lasso_model.C = best_c
lasso_model.fit(X_train_resampled, y_train_resampled)

# Make predictions
y_pred = lasso_model.predict(X_test)


def roc(X_test,y_test, model):
    # ROC Curve----------------------------------------------------------------------------------------
    # Assuming you have already trained your logistic regression model and obtained predicted probabilities
    # y_true: True labels of the testing dataset (ground truth)
    # y_prob: Predicted probabilities for the positive class (class 1) from your logistic regression model

    y_prob = model.predict_proba(X_test)[:, 1]
    # Calculate the false positive rate (FPR), true positive rate (TPR), and threshold values for the ROC curve
    fpr, tpr, thresholds = roc_curve(y_test, y_prob)

    # Calculate the area under the ROC curve
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
    plt.figure(figsize=[10, 8])
    norm_cm = cm
    if normalized:
        norm_cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        sns.heatmap(norm_cm, annot=cm, fmt='g', xticklabels=classes, yticklabels=classes, cmap=cmap)
    plt.show()



# Evaluate the model's performance
accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred, zero_division=1)
recall = recall_score(y_test, y_pred, zero_division=1)
f1 = f1_score(y_test, y_pred, zero_division=1)

print(f"Accuracy: {accuracy:.2f}")
print(f"Precision: {precision:.2f}")
print(f"Recall: {recall:.2f}")
print(f"F1-score: {f1:.2f}")

roc(X_test,y_test,lasso_model)
plot_confusion_matrix(confusion_matrix(y_test,y_pred), ['0', '1'])