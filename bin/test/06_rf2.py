import numpy as np
from sklearn.metrics import classification_report
import joblib

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

test_file_path = '../../data/test/miR_T.txt'
model_file_path = '../../models_results/animal_plant/best_random_forest_model_5.joblib'
X_test, y_test = load_data_v3(test_file_path)

model = joblib.load(model_file_path)

print("Testing...")
y_pred = model.predict(X_test)

# Evaluate the model and write the results to a file.

print("Test Set Classification Report:\n")
print(classification_report(y_test, y_pred, digits=5))
