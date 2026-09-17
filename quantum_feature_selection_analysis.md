# Feature Selection Comparison for the Fraud Detection Project

## 1. Algorithms Included

This project compares three feature-selection approaches:

1. SelectKBest (classical baseline)
2. Quantum-inspired genetic feature selection
3. Quantum-inspired differential evolution (QDE)

The comparison is designed to answer the boss requirement of using a normal classical method first, then two quantum-inspired methods, and then comparing their behavior and usefulness for this fraud-detection system.

---

## 2. Parameter Setup Used for Comparison

The project now records comparison metrics for each method in the feature-selection layer.

### SelectKBest
- Method: `f_classif`
- Parameter: `k = min(20, n_features)`
- Evaluation: 3-fold cross-validation using ROC-AUC
- Best use: fast feature ranking when features are already mostly informative

### Quantum Genetic Feature Selection
- Population size: `min(40, max(10, 6 * sqrt(n_features)))`
- Maximum iterations: `30`
- Mutation rate: `0.01`
- Evaluation: 3-fold ROC-AUC using logistic regression
- Best use: lightweight global search when the feature set is noisy but not too large

### Quantum-Inspired Differential Evolution (QDE)
- Population size: `12`
- Maximum iterations: `15`
- Mutation factor: `0.8`
- Crossover rate: `0.9`
- Target subset size: `min(20, n_features)`
- Evaluation: 3-fold ROC-AUC using logistic regression
- Best use: stronger global search for subset optimization when feature interactions matter

These parameters are intentionally simple and project-friendly. They are easy to tune later once the real dataset and validation results are available.

---

## 3. How Each Algorithm Works

### A. SelectKBest
SelectKBest is a classical filter method. It scores each feature independently using a statistical test such as ANOVA F-score and keeps the top `k` features.

Pros:
- very fast
- simple to understand and explain
- low computational cost
- good baseline for comparison

Cons:
- assumes each feature matters independently
- ignores feature interaction effects
- may keep redundant or weakly useful features together
- can miss important combinations of features

For this project, it is a good baseline because it is simple, transparent, and easy for a boss to understand.

### B. Quantum-Inspired Genetic Feature Selection
This method uses a probability vector that acts like a quantum-inspired representation of whether a feature is likely to be selected. It samples many binary feature subsets, tests them, and updates the probability vector toward the best-performing subset.

Pros:
- good at exploring feature subsets globally
- better than independent ranking when feature interaction exists
- more adaptive than plain filter methods
- naturally supports probabilistic search behavior

Cons:
- slower than SelectKBest
- may converge to a local optimum if not tuned well
- results can vary more because of randomness
- less stable than a deterministic ranking method

For this fraud dataset, this approach is sensible because fraud patterns often arise from subtle combinations of variables rather than one obvious feature alone.

### C. Quantum-Inspired Differential Evolution (QDE)
QDE evolves a population of candidate feature subsets using mutation, crossover, and selection. The quantum-inspired part is usually introduced through probability-based updates or quantum-style search behavior to improve exploration and avoid premature convergence.

Pros:
- strong global search capability
- good at finding useful combinations of features
- often better than simple genetic search when the search space is complex
- can be effective for high-dimensional and noisy datasets

Cons:
- more computationally demanding than SelectKBest
- more tuning effort is needed
- harder to explain to non-technical stakeholders
- random behavior requires careful validation

For a fraud detection project, this is attractive because the dataset is high-dimensional and the fraud signal is often hidden in nonlinear feature relationships.

---

## 4. Pros and Cons Relative to This Project

### SelectKBest
Pros:
- fastest and easiest to interpret
- ideal as the control method
- good for quick benchmark comparison

Cons:
- likely weakest for complex fraud patterns
- may miss highly informative feature combinations
- not a true optimizer for subset selection

### Quantum-Inspired Genetic Selection
Pros:
- very suitable for this project structure
- better candidate subset exploration than SelectKBest
- fits the existing Layer 3 optimization architecture

Cons:
- not always as stable as classical methods
- tuning is necessary
- can be slower than a filter approach

### QDE
Pros:
- strongest global optimization behavior among the three
- likely best for noisy, nonlinear, interaction-heavy fraud data
- aligns with the “quantum-inspired” requirement from the boss

Cons:
- more complex to implement and tune
- may take longer than the existing genetic method
- may not be worth the overhead if the dataset is too small or too simple

---

## 5. Conclusion for This Project Scope

For this fraud-detection system, the best practical interpretation is:

- SelectKBest is the baseline and reference method.
- The quantum-inspired genetic selector is a realistic middle ground that matches the current project architecture and is easier to integrate.
- QDE is the more advanced optimizer and is likely the strongest candidate when the goal is to improve global feature subset optimization and capture complex fraud patterns.

The likely outcome is:
- SelectKBest will be the fastest and easiest to explain.
- Quantum-inspired genetic selection will probably improve over SelectKBest in terms of feature subset quality but may vary more.
- QDE may produce the best feature subset quality if tuned properly, especially when feature interactions drive fraud detection performance.

Because this is a fraud dataset with many correlated variables and a strong class imbalance, I would expect QDE to be the most promising if the project wants a true quantum-inspired optimization layer that goes beyond simple ranking. However, in a real production implementation, the best choice would be based on validation metrics such as ROC-AUC, recall, F1-score, and runtime.

## Final Judgment

For the scope of this project, using all three methods and comparing them is the right approach. It satisfies the boss request, keeps the baseline classical method, and demonstrates two quantum-inspired alternatives in a clear, defensible way. QDE is especially useful as the advanced method, while SelectKBest remains the benchmark and the quantum genetic approach is the simpler in-project optimization method.
