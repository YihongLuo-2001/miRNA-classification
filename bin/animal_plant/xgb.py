import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import classification_report
from joblib import dump
import xgboost as xgb


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
train_file_path = '../../data/animal_plant_dataset/rna_train.txt'
test_file_path = '../../data/animal_plant_dataset/rna_test.txt'
output_file_path = '../../models_results/animal_plant/xgb_results.txt'
model_file_path = '../../models_results/animal_plant/xgb_model_5.joblib'

print("Processing data...")
X_train, y_train = load_data_v3(train_file_path)
X_test, y_test = load_data_v3(test_file_path)

param_grid = {
    # 'n_estimators': [50, 100, 200],
    'max_depth': [20, 30, 40, None],
    'learning_rate': [0.01, 0.1, 0.2],
    'subsample': [0.8, 1.0],
    'colsample_bytree': [0.8, 1.0]
}

print("Initialize the Random Forest classifier...")
xgb_classifier = xgb.XGBClassifier(random_state=42, eval_metric='logloss')

# Create a GridSearchCV instance.
print("Start cross-validation to find the optimal hyperparameters...")

# n_jobs(Threads)=20
grid_search = GridSearchCV(estimator=xgb_classifier, param_grid=param_grid, cv=5, verbose=3, n_jobs=20)

# Perform cross-validation on the training data.
grid_search.fit(np.array(X_train).reshape(len(X_train), -1), y_train)
print("Cross-validation completed! ")

print("Use the optimal hyperparameters for prediction...")
best_model = grid_search.best_estimator_
print('Best model: ', best_model)

# Training model with the optimal hyperparameters for prediction.
print("Training Random Forest with the optimal hyperparameters...")
best_model.fit(X_train, y_train)
print("Training completed!")

# Use the model with the optimal hyperparameters for prediction.
y_pred_test = best_model.predict(np.array(X_test).reshape(len(X_test), -1))

# Evaluate the model and write the results to a file.
with open(output_file_path, 'w') as f:
    f.write("Best Hyperparameters:\n")
    f.write(str(grid_search.best_params_) + "\n\n")
    f.write("Test Set Classification Report:\n")
    f.write(classification_report(y_test, y_pred_test, digits=5))
print("The evaluation results have been written to the file!")

# Save the trained optimal model.
dump(best_model, model_file_path)
print(f"The optimal model has been saved to {model_file_path}")
