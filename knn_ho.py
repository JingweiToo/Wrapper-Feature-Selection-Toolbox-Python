from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.neighbors import KNeighborsClassifier

from niapy.problems import Problem
from niapy.task import OptimizationType, Task
from niapy.algorithms.modified import HybridBatAlgorithm


def get_hyperparameters(x):
    """Get hyperparameters for solution `x`."""
    algorithms = ('ball_tree', 'kd_tree', 'brute')
    n_neighbors = int(5 + x[0] * 10)
    weights = 'uniform' if x[1] < 0.5 else 'distance'
    algorithm = algorithms[int(x[2] * 2)]
    leaf_size = int(10 + x[3] * 40)

    params =  {
        'n_neighbors': n_neighbors,
        'weights': weights,
        'algorithm': algorithm,
        'leaf_size': leaf_size
    }
    return params


def get_classifier(x):
    """Get classifier from solution `x`."""
    params = get_hyperparameters(x)
    return KNeighborsClassifier(**params)

class KNNHyperparameterOptimization(Problem):
    def __init__(self, X_train, y_train):
        super().__init__(dimension=4, lower=0, upper=1)
        self.X_train = X_train
        self.y_train = y_train

    def _evaluate(self, x):
        model = get_classifier(x)
        scores = cross_val_score(model, self.X_train, self.y_train, cv=2, n_jobs=-1)
        return scores.mean()

# train and test    
X, y = load_breast_cancer(return_X_y=True)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=1234)

# run algorithm
problem = KNNHyperparameterOptimization(X_train, y_train)

# We will be running maximization for 100 iters on `problem`
task = Task(problem, max_iters=100, optimization_type=OptimizationType.MAXIMIZATION)

algorithm = HybridBatAlgorithm(population_size=10, seed=1234)
best_params, best_accuracy = algorithm.run(task)

print('Best parameters:', get_hyperparameters(best_params))

# compare our optimal model with the default one

default_model = KNeighborsClassifier()
best_model = get_classifier(best_params)

default_model.fit(X_train, y_train)
best_model.fit(X_train, y_train)

default_score = default_model.score(X_test, y_test)
best_score = best_model.score(X_test, y_test)

print('Default model accuracy:', default_score)
print('Best model accuracy:', best_score)