import numpy as np
from sklearn.svm import SVC
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import classification_report
from joblib import dump

np.random.seed(0)


def load_data_v3(file_path):
    """Load the data according to the new file format and separate the features and labels. """
    print(f"Loading data...：{file_path}")
    with open(file_path, 'r') as file:
        content = file.read()

    parts = content.strip().split('>')[1:]
    X, y = [], []

    for part in parts:
        lines = part.strip().split('\n')
        label = int(lines[0].split('\t')[-1])
        y.append(label)

        # Every four lines represent the complete encoding of a nucleotide sequence, processed column by column.
        one_hot_encoded = np.array([list(map(int, line)) for line in lines[2:6]]).T
        X.append(one_hot_encoded)

    print(
        f"Complete loading! Data set: {len(X)} samples, each sample has {len(X[0])} types of nucleotide bases, the vector length for each nucleotide base is {len(X[0][0])}")
    return np.array(X).reshape(len(X), -1), np.array(y)


# Path
train_file_path = '../../data/families_20_dataset/set_train_families-filtered.txt'
test_file_path = '../../data/families_20_dataset/set_test_families-filtered.txt'
model_file_path = '../../models_results/families_20/svm_model_6_families.joblib'
evaluation_report_path = '../../models_results/families_20/SVM_report_6_families.txt'

print("Processing data...")
X_train, y_train = load_data_v3(train_file_path)
X_test, y_test = load_data_v3(test_file_path)

param_grid = {'kernel': ['linear', 'poly', 'rbf', 'sigmoid'],
              'C': [0.5, 1, 5, 10]}

print("Initialize the Support Vector Machine classifier...")
svm_model = SVC(random_state=42)
# Create a GridSearchCV instance.
print("Start cross-validation to find the optimal hyperparameters...")
grid_search = GridSearchCV(estimator=svm_model, param_grid=param_grid, cv=5, verbose=3)

# Perform cross-validation on the training data.
grid_search.fit(np.array(X_train).reshape(len(X_train), -1), y_train)
print("Cross-validation completed! ")

# Use the model with the optimal hyperparameters for prediction.
print("Use the optimal hyperparameters for prediction...")
best_model = grid_search.best_estimator_
print('Best model: ', best_model)

# Training model with the optimal hyperparameters for prediction.
print("Training Support Vector Machine with the optimal hyperparameters...")
best_model.fit(X_train, y_train)
print("Training completed!")

print("Testing...")
y_pred = best_model.predict(X_test)

# Evaluate the model and write the results to a file.
with open(evaluation_report_path, 'w') as file:
    file.write("Best Hyperparameters:\n")
    file.write(str(grid_search.best_params_) + "\n\n")
    file.write("Test Set Classification Report:\n")
    file.write(classification_report(y_test, y_pred, digits=5))
print("The evaluation results have been written to the file!")

# Save the trained optimal model.
dump(best_model, model_file_path)
print(f"The optimal model has been saved to {model_file_path}")
