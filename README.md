# From Scratch
A collection of algorithms and experiments touching various areas of Machine Learning and its related fields. 
Implemented in Python from scratch targeting in minimal and clean code.

## Implementations
- Supervised Learning
    - Multi-layer Neural Networks [[demo]](./examples/supervised/backpropagation.ipynb) [[code]](./fromscratch/supervised/neuralnetworks)
    - Classification Trees [[demo]](./examples/supervised/classification-tree.ipynb) [[code]](./fromscratch/supervised/classification_tree.py)
    - Adaptive Boosting [[demo]](./examples/supervised/adaboost.ipynb) [[code]](./fromscratch/supervised/adaboost.py)
    - Support Vector Machine [[demo]](./examples/supervised/svm-classifier.ipynb) [[code]](./fromscratch/supervised/svm_classifier.py)
    - Linear Regression [[demo]](./examples/supervised/linear-regression.ipynb) [[code]](./fromscratch/supervised/linear_regression.py)

- Unsupervised Learning
    - K-means [[demo]](./examples/unsupervised/kmeans.ipynb) [[code]](./fromscratch/unsupervised/kmeans.py)
    - DBSCAN [[demo]](./examples/unsupervised/dbscan.ipynb) [[code]](./fromscratch/unsupervised/dbscan.py)

- Reinforcement Learning
    - N-armed Bandits [[demo]](./examples/rl/n-armed-bandit.ipynb) [[code]](./fromscratch/rl/bandit.py)
    - Dynamic Programming [[demo]](./examples/rl/policy-iteration.ipynb) [[code]](./fromscratch/rl/dp.py)
    - SARSA [[demo]](./examples/rl/temporal-difference-learning.ipynb) [[code]](./fromscratch/rl/td_learning.py)
    - Q-Learning [[demo]](./examples/rl/temporal-difference-learning.ipynb) [[code]](./fromscratch/rl/td_learning.py)

## Dependencies
- **numpy**: Used in all implementations for vector/matrix operations and vectorized calculations
- **cvxopt**: Used in SVM for solving the quadratic programming problem
- **scipy**: Borrowed its KDTree implementation for fast nearest neighbours calculation
