import numpy as np
import pandas as pd

class AdaBoost:
    def __init__(self, n_estimators=50):
        self.n_estimators = n_estimators
        self.models = []
        self.model_weights = []

    def fit(self, X, y):
        n_samples = X.shape[0]
        weights = np.ones(n_samples) / n_samples

        for _ in range(self.n_estimators):
            model = DecisionStump()
            model.fit(X, y, weights)
            predictions = model.predict(X)

            error = np.sum(weights * (predictions != y)) / np.sum(weights)
            model_weight = 0.5 * np.log((1 - error) / (error + 1e-10))

            weights *= np.exp(-model_weight * y * predictions)
            weights /= np.sum(weights)

            self.models.append(model)
            self.model_weights.append(model_weight)

    def predict(self, X):
        final_predictions = np.zeros(X.shape[0])
        for model, weight in zip(self.models, self.model_weights):
            predictions = model.predict(X)
            final_predictions += weight * predictions
        return np.sign(final_predictions)

class DecisionStump:
    def __init__(self):
        self.feature_index = None
        self.threshold = None
        self.polarity = 1

    def fit(self, X, y, weights):
        m, n = X.shape
        min_error = float('inf')

        for feature_index in range(n):
            thresholds = np.unique(X[:, feature_index])
            for threshold in thresholds:
                for polarity in [1, -1]:
                    predictions = np.ones(m)
                    predictions[X[:, feature_index] < threshold] = -1
                    predictions *= polarity

                    error = np.sum(weights * (predictions != y))

                    if error < min_error:
                        min_error = error
                        self.feature_index = feature_index
                        self.threshold = threshold
                        self.polarity = polarity

    def predict(self, X):
        n_samples = X.shape[0]
        predictions = np.ones(n_samples)
        predictions[X[:, self.feature_index] < self.threshold] = -1
        return predictions * self.polarity

# Example dataset
data = {
    'feature1': [1, 2, 3, 4, 5],
    'feature2': [2, 3, 4, 5, 6],
    'target': [1, 1, -1, -1, 1]
}
df = pd.DataFrame(data)
X = df[['feature1', 'feature2']].values
y = df['target'].values

# Create and train the model
ada_model = AdaBoost(n_estimators=5)
ada_model.fit(X, y)

# Make predictions
predictions = ada_model.predict(X)
print("Predictions:", predictions)