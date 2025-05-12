import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.svm import SVC 
from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score, f1_score, classification_report

url = "https://raw.githubusercontent.com/Agradoot1/customer-churn/main/customer.xlsx"

df = pd.read_excel(url, sheet_name='Data for DSBA')

# Display the first few rows of the DataFrame
print(df)

target_column = 'Churn'  # Replace with the actual column name
X = df.drop(columns=[target_column])  # X contains all columns except the target column
y = df['Churn']  # y contains the target column

# Define numerical columns based on their data types
numerical_cols = X.select_dtypes(include=['int64', 'float64']).columns

# Identify categorical columns by excluding numerical columns
categorical_cols = X.columns.difference(numerical_cols)

# Convert all categorical columns to strings
X[categorical_cols] = X[categorical_cols].astype(str)

# Split data into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Preprocessing pipeline for numerical columns
numerical_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='mean')),  # Impute missing values with mean
    ('scaler', StandardScaler())  # Standardize numerical features
])

# Preprocessing pipeline for categorical columns
categorical_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='most_frequent')),  # Impute missing values with most frequent value
    ('onehot', OneHotEncoder(handle_unknown='ignore'))  # One-hot encode categorical features
])

# Combine preprocessing steps for numerical and categorical columns
preprocessor = ColumnTransformer(transformers=[
    ('num', numerical_transformer, numerical_cols),
    ('cat', categorical_transformer, categorical_cols)
])

# Display column names with their data types
for column in X.columns:
    print(f"Column: {column}, Dtype: {X[column].dtype}")

# Define the logistic regression model pipeline
model = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('classifier', LogisticRegression())  # Logistic Regression classifier
])

# Train the model (includes preprocessing)
model.fit(X_train, y_train)

# Make predictions on the test set
y_pred = model.predict(X_test)
y_pred_train = model.predict(X_train)

# Compute confusion matrix
cm = confusion_matrix(y_test, y_pred)

# Calculate accuracy
accuracy = accuracy_score(y_test, y_pred)

# Calculate sensitivity (recall) and specificity
tn, fp, fn, tp = cm.ravel()

sensitivity = tp / (tp + fn)  # True Positive Rate (Recall)
specificity = tn / (tn + fp)  # True Negative Rate

# Calculate precision, recall, and F1-score
precision = precision_score(y_test, y_pred)
recall = recall_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)

# Define the Random Forest model pipeline
rf_model = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('classifier', RandomForestClassifier(random_state=42))  # Random Forest classifier
])

# Train the Random Forest model (includes preprocessing)
rf_model.fit(X_train, y_train)

# Make predictions on the test set
y_pred_rf = rf_model.predict(X_test)
y_pred_train_rf = rf_model.predict(X_train)

# Compute confusion matrix and accuracy for Random Forest
cm_rf = confusion_matrix(y_test, y_pred_rf)
accuracy_rf = accuracy_score(y_test, y_pred_rf)

# Calculate sensitivity (recall) and specificity
tn, fp, fn, tp = cm.ravel()

sensitivity = tp / (tp + fn)  # True Positive Rate (Recall)
specificity = tn / (tn + fp)  # True Negative Rate

# Calculate precision, recall, and F1-score
precision = precision_score(y_test, y_pred)
recall = recall_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)

# Define the Support Vector Machine (SVM) model pipeline
svm_model = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('classifier', SVC(kernel='rbf', random_state=42))  # SVM classifier with RBF kernel
])

# Train the SVM model (includes preprocessing)
svm_model.fit(X_train, y_train)

# Make predictions on the test set
y_pred_svm = svm_model.predict(X_test)
y_pred_train_svm = model.predict(X_train)
# Compute confusion matrix and accuracy for SVM
cm_svm = confusion_matrix(y_test, y_pred_svm)
accuracy_svm = accuracy_score(y_test, y_pred_svm)
# Calculate sensitivity (recall) and specificity
tn, fp, fn, tp = cm.ravel()

sensitivity = tp / (tp + fn)  # True Positive Rate (Recall)
specificity = tn / (tn + fp)  # True Negative Rate

# Calculate precision, recall, and F1-score
precision = precision_score(y_test, y_pred)
recall = recall_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)

# Define the Decision Tree model pipeline
dt_model = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('classifier', DecisionTreeClassifier(random_state=42))  # Decision Tree classifier
])

# Train the Decision Tree model (includes preprocessing)
dt_model.fit(X_train, y_train)

# Make predictions on the test set
y_pred_dt = dt_model.predict(X_test)

# Compute confusion matrix and accuracy for Decision Tree
cm_dt = confusion_matrix(y_test, y_pred_dt)
accuracy_dt = accuracy_score(y_test, y_pred_dt)
# Calculate sensitivity (recall) and specificity
tn, fp, fn, tp = cm.ravel()

sensitivity = tp / (tp + fn)  # True Positive Rate (Recall)
specificity = tn / (tn + fp)  # True Negative Rate

# Calculate precision, recall, and F1-score
precision = precision_score(y_test, y_pred)
recall = recall_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)

# Define a function to evaluate a model
def evaluate_model(model, X_train, X_test, y_train, y_test):
    # Train the model
    model.fit(X_train, y_train)
    
    # Make predictions
    y_pred_train = model.predict(X_train)
    y_pred_test = model.predict(X_test)
    
    # Calculate metrics for train set
    accuracy_train = accuracy_score(y_train, y_pred_train)
    precision_train = precision_score(y_train, y_pred_train)
    recall_train = recall_score(y_train, y_pred_train)
    f1_train = f1_score(y_train, y_pred_train)
    
    # Calculate confusion matrix for train set
    cm_train = confusion_matrix(y_train, y_pred_train)
    tn_train, fp_train, fn_train, tp_train = cm_train.ravel()
    
    # Calculate metrics for test set
    accuracy_test = accuracy_score(y_test, y_pred_test)
    precision_test = precision_score(y_test, y_pred_test)
    recall_test = recall_score(y_test, y_pred_test)
    f1_test = f1_score(y_test, y_pred_test)
    
    # Calculate confusion matrix for test set
    cm_test = confusion_matrix(y_test, y_pred_test)
    tn_test, fp_test, fn_test, tp_test = cm_test.ravel()
    
    # Calculate specificity for train and test sets
    specificity_train = tn_train / (tn_train + fp_train)
    specificity_test = tn_test / (tn_test + fp_test)
    
    # Print the metrics
    print("Metrics for Training Set:")
    print(f"Accuracy: {accuracy_train:.4f}")
    print(f"Precision: {precision_train:.4f}")
    print(f"Recall (Sensitivity): {recall_train:.4f}")
    print(f"F1-Score: {f1_train:.4f}")
    print(f"Specificity: {specificity_train:.4f}")
    print("Confusion Matrix:")
    print(cm_train)

    print("\nMetrics for Test Set:")
    print(f"Accuracy: {accuracy_test:.4f}")
    print(f"Precision: {precision_test:.4f}")
    print(f"Recall (Sensitivity): {recall_test:.4f}")
    print(f"F1-Score: {f1_test:.4f}")
    print(f"Specificity: {specificity_test:.4f}")
    print("Confusion Matrix:")
    print(cm_test)

# Assuming you have defined your models (e.g., logistic_model, rf_model, svm_model)

# Evaluate each model using the defined function
print("Evaluation for Logistic Regression Model:")
evaluate_model( model , X_train, X_test, y_train, y_test)



print("\nEvaluation for Random Forest Model:")
evaluate_model(rf_model, X_train, X_test, y_train, y_test)



print("\nEvaluation for Support Vector Machine (SVM) Model:")
evaluate_model(svm_model, X_train, X_test, y_train, y_test)


print("Evaluation for decision tree Model:")
evaluate_model( dt_model , X_train, X_test, y_train, y_test)

class_report = classification_report(y_test, y_pred)
print('Classification Report:')
print(class_report)

# Handle non-numeric values in numerical columns
for col in numerical_cols:
    df[col] = pd.to_numeric(df[col], errors='coerce')

# Remove rows with NaN values
df = df.dropna()

# Correlation Matrix Heatmap for Numerical Columns
plt.figure(figsize=(10, 8))
correlation_matrix = df[numerical_cols].corr()
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', center=0)
plt.title('Correlation Matrix Heatmap for Numerical Variables')
plt.show()

for col in categorical_cols:
    plt.figure(figsize=(10, 6))
    sns.countplot(data=df, x=col, hue='Churn', palette='coolwarm')
    plt.title(f'{col} vs Churn')
    plt.xlabel(col)
    plt.ylabel('Count')
    plt.legend(title='Churn')
    plt.show()