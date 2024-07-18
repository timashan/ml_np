import numpy as np


class NaiveBayes:
    def fit(self, X, y):
        n_samples, n_features = X.shape
        self._classes = np.unique(y)
        n_classes = len(self._classes)

        self._mean = np.zeros((n_classes, n_features), dtype=np.float64)
        self._var = np.zeros((n_classes, n_features), dtype=np.float64)
        self._priors = np.zeros(n_classes, dtype=np.float64)

        for i, c in enumerate(self._classes):
            # Features for each class
            X_c = X[y == c]
            # Mean & Variance for each feature
            self._mean[i, :] = X_c.mean(axis=0)
            self._var[i, :] = X_c.var(axis=0)
            # Relative frequencies of classes as Priors
            self._priors[i] = X_c.shape[0] / n_samples

    def predict(self, X):
        return [self._predict(x) for x in X]

    def _predict(self, x):
        poss = []
        for i, c in enumerate(self._classes):
            prior = np.log(self._priors[i])
            pos = np.sum(np.log(self._pdf(i, x)))
            # Logs so we sum em up
            pos = pos + prior
            poss.append(pos)
        return self._classes[np.argmax(poss)]

    def _pdf(self, i, x):
        # Returns list of pdfs for each feature
        mean = self._mean[i]
        var = self._var[i]
        numerator = np.exp(-((x - mean) ** 2) / (2 * var))
        denominator = np.sqrt(2 * np.pi * var)
        return numerator / denominator
