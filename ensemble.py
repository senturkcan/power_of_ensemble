#take best hyperparameters for models. Make the best ensemble. Evaluate at the same time for choosing.
#choose how the models can be evaluated to be the "best"

#dt
#knn
#svm

"""
Comprehensive hyperparameter tuning for multiple classifiers
with ensemble methods (Hard Voting & Stacking)
"""

import optuna
import numpy as np
from sklearn.datasets import load_wine
from sklearn.model_selection import cross_val_score, cross_val_predict, StratifiedKFold, train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier, VotingClassifier, StackingClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import RidgeClassifier, LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.neural_network import MLPClassifier

# Load dataset
X, y = load_wine(return_X_y=True)

# Split data for final evaluation
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# Define CV strategy (on training data)
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

# Suppress optuna logging for cleaner output
optuna.logging.set_verbosity(optuna.logging.WARNING)


# ---------- 1️⃣ Decision Tree ----------
def objective_decision_tree(trial):
    criterion = trial.suggest_categorical("criterion", ["gini", "entropy", "log_loss"])
    max_depth = trial.suggest_int("max_depth", 2, 20)
    min_samples_split = trial.suggest_int("min_samples_split", 2, 10)
    min_samples_leaf = trial.suggest_int("min_samples_leaf", 1, 5)
    
    model = DecisionTreeClassifier(
        criterion=criterion,
        max_depth=max_depth,
        min_samples_split=min_samples_split,
        min_samples_leaf=min_samples_leaf,
        random_state=42
    )
    
    score = cross_val_score(model, X_train, y_train, cv=cv, scoring="accuracy").mean()
    return score


# ---------- 2️⃣ K-Nearest Neighbors ----------
def objective_knn(trial):
    n_neighbors = trial.suggest_int("n_neighbors", 1, 20)
    weights = trial.suggest_categorical("weights", ["uniform", "distance"])
    p = trial.suggest_int("p", 1, 2)
    
    model = Pipeline([
        ("scaler", StandardScaler()),
        ("knn", KNeighborsClassifier(n_neighbors=n_neighbors, weights=weights, p=p))
    ])
    
    score = cross_val_score(model, X_train, y_train, cv=cv, scoring="accuracy").mean()
    return score


# ---------- 3️⃣ Support Vector Classifier ----------
def objective_svc(trial):
    C = trial.suggest_float("C", 1e-3, 1e3, log=True)
    kernel = trial.suggest_categorical("kernel", ["linear", "poly", "rbf", "sigmoid"])
    gamma = trial.suggest_categorical("gamma", ["scale", "auto"])
    
    model = Pipeline([
        ("scaler", StandardScaler()),
        ("svc", SVC(C=C, kernel=kernel, gamma=gamma, random_state=42))
    ])
    
    score = cross_val_score(model, X_train, y_train, cv=cv, scoring="accuracy").mean()
    return score


# ---------- 4️⃣ Random Forest ----------
def objective_rf(trial):
    n_estimators = trial.suggest_int("n_estimators", 50, 300)
    max_depth = trial.suggest_int("max_depth", 2, 20)
    min_samples_split = trial.suggest_int("min_samples_split", 2, 10)
    min_samples_leaf = trial.suggest_int("min_samples_leaf", 1, 5)
    max_features = trial.suggest_categorical("max_features", ["sqrt", "log2", None])
    
    model = RandomForestClassifier(
        n_estimators=n_estimators,
        max_depth=max_depth,
        min_samples_split=min_samples_split,
        min_samples_leaf=min_samples_leaf,
        max_features=max_features,
        random_state=42
    )
    
    score = cross_val_score(model, X_train, y_train, cv=cv, scoring="accuracy").mean()
    return score


# ---------- 5️⃣ Naive Bayes ----------
def objective_nb(trial):
    var_smoothing = trial.suggest_float("var_smoothing", 1e-10, 1e-5, log=True)
    
    model = Pipeline([
        ("scaler", StandardScaler()),
        ("nb", GaussianNB(var_smoothing=var_smoothing))
    ])
    
    score = cross_val_score(model, X_train, y_train, cv=cv, scoring="accuracy").mean()
    return score


# ---------- 6️⃣ Ridge Classifier ----------
def objective_ridge(trial):
    alpha = trial.suggest_float("alpha", 1e-3, 1e3, log=True)
    
    model = Pipeline([
        ("scaler", StandardScaler()),
        ("ridge", RidgeClassifier(alpha=alpha, random_state=42))
    ])
    
    score = cross_val_score(model, X_train, y_train, cv=cv, scoring="accuracy").mean()
    return score


# ---------- 7️⃣ Logistic Regression ----------
def objective_lr(trial):
    C = trial.suggest_float("C", 1e-3, 1e3, log=True)
    solver = trial.suggest_categorical("solver", ["lbfgs", "saga"])
    
    model = Pipeline([
        ("scaler", StandardScaler()),
        ("lr", LogisticRegression(C=C, solver=solver, max_iter=1000, random_state=42))
    ])
    
    score = cross_val_score(model, X_train, y_train, cv=cv, scoring="accuracy").mean()
    return score


# ---------- 8️⃣ Deep Learning (MLP) - First Model ----------
def objective_mlp1(trial):
    hidden_layer_sizes = trial.suggest_categorical("hidden_layer_sizes", 
                                                   [(50,), (100,), (50, 50), (100, 50)])
    activation = trial.suggest_categorical("activation", ["relu", "tanh"])
    alpha = trial.suggest_float("alpha", 1e-5, 1e-2, log=True)
    learning_rate_init = trial.suggest_float("learning_rate_init", 1e-4, 1e-2, log=True)
    
    model = Pipeline([
        ("scaler", StandardScaler()),
        ("mlp", MLPClassifier(
            hidden_layer_sizes=hidden_layer_sizes,
            activation=activation,
            alpha=alpha,
            learning_rate_init=learning_rate_init,
            max_iter=500,
            random_state=42,
            early_stopping=True
        ))
    ])
    
    score = cross_val_score(model, X_train, y_train, cv=cv, scoring="accuracy").mean()
    return score


# ---------- 9️⃣ Deep Learning (MLP) - Second Model ----------
def objective_mlp2(trial):
    hidden_layer_sizes = trial.suggest_categorical("hidden_layer_sizes", 
                                                   [(100, 100), (100, 50, 25), (150, 75)])
    activation = trial.suggest_categorical("activation", ["relu", "logistic"])
    alpha = trial.suggest_float("alpha", 1e-5, 1e-2, log=True)
    learning_rate = trial.suggest_categorical("learning_rate", ["constant", "adaptive"])
    
    model = Pipeline([
        ("scaler", StandardScaler()),
        ("mlp", MLPClassifier(
            hidden_layer_sizes=hidden_layer_sizes,
            activation=activation,
            alpha=alpha,
            learning_rate=learning_rate,
            max_iter=500,
            random_state=42,
            early_stopping=True
        ))
    ])
    
    score = cross_val_score(model, X_train, y_train, cv=cv, scoring="accuracy").mean()
    return score


# ---------- Run hyperparameter tuning ----------
print("=" * 60)
print("HYPERPARAMETER TUNING")
print("=" * 60)

studies = {}

print("\n🌳 Tuning Decision Tree...")
studies['dt'] = optuna.create_study(direction="maximize")
studies['dt'].optimize(objective_decision_tree, n_trials=30, show_progress_bar=True)
print(f"Best accuracy: {studies['dt'].best_value:.4f}")

print("\n👥 Tuning KNN...")
studies['knn'] = optuna.create_study(direction="maximize")
studies['knn'].optimize(objective_knn, n_trials=30, show_progress_bar=True)
print(f"Best accuracy: {studies['knn'].best_value:.4f}")

print("\n🎯 Tuning SVC...")
studies['svc'] = optuna.create_study(direction="maximize")
studies['svc'].optimize(objective_svc, n_trials=30, show_progress_bar=True)
print(f"Best accuracy: {studies['svc'].best_value:.4f}")

print("\n🌲 Tuning Random Forest...")
studies['rf'] = optuna.create_study(direction="maximize")
studies['rf'].optimize(objective_rf, n_trials=30, show_progress_bar=True)
print(f"Best accuracy: {studies['rf'].best_value:.4f}")

print("\n📊 Tuning Naive Bayes...")
studies['nb'] = optuna.create_study(direction="maximize")
studies['nb'].optimize(objective_nb, n_trials=20, show_progress_bar=True)
print(f"Best accuracy: {studies['nb'].best_value:.4f}")

print("\n📏 Tuning Ridge Classifier...")
studies['ridge'] = optuna.create_study(direction="maximize")
studies['ridge'].optimize(objective_ridge, n_trials=20, show_progress_bar=True)
print(f"Best accuracy: {studies['ridge'].best_value:.4f}")

print("\n📈 Tuning Logistic Regression...")
studies['lr'] = optuna.create_study(direction="maximize")
studies['lr'].optimize(objective_lr, n_trials=20, show_progress_bar=True)
print(f"Best accuracy: {studies['lr'].best_value:.4f}")

print("\n🧠 Tuning Deep Learning Model 1...")
studies['mlp1'] = optuna.create_study(direction="maximize")
studies['mlp1'].optimize(objective_mlp1, n_trials=30, show_progress_bar=True)
print(f"Best accuracy: {studies['mlp1'].best_value:.4f}")

print("\n🧠 Tuning Deep Learning Model 2...")
studies['mlp2'] = optuna.create_study(direction="maximize")
studies['mlp2'].optimize(objective_mlp2, n_trials=30, show_progress_bar=True)
print(f"Best accuracy: {studies['mlp2'].best_value:.4f}")


# ---------- Build best models ----------
best_dt = DecisionTreeClassifier(**studies['dt'].best_params, random_state=42)

best_knn = Pipeline([
    ("scaler", StandardScaler()),
    ("knn", KNeighborsClassifier(**studies['knn'].best_params))
])

best_svc = Pipeline([
    ("scaler", StandardScaler()),
    ("svc", SVC(**studies['svc'].best_params, random_state=42))
])

best_rf = RandomForestClassifier(**studies['rf'].best_params, random_state=42)

best_nb = Pipeline([
    ("scaler", StandardScaler()),
    ("nb", GaussianNB(**studies['nb'].best_params))
])

best_ridge = Pipeline([
    ("scaler", StandardScaler()),
    ("ridge", RidgeClassifier(**studies['ridge'].best_params, random_state=42))
])

best_lr = Pipeline([
    ("scaler", StandardScaler()),
    ("lr", LogisticRegression(**studies['lr'].best_params, max_iter=1000, random_state=42))
])

best_mlp1 = Pipeline([
    ("scaler", StandardScaler()),
    ("mlp", MLPClassifier(**studies['mlp1'].best_params, max_iter=500, random_state=42, early_stopping=True))
])

best_mlp2 = Pipeline([
    ("scaler", StandardScaler()),
    ("mlp", MLPClassifier(**studies['mlp2'].best_params, max_iter=500, random_state=42, early_stopping=True))
])


# ---------- Hard Voting Ensemble (with 2 best DL models) ----------
print("\n" + "=" * 60)
print("ENSEMBLE METHODS")
print("=" * 60)

voting_clf = VotingClassifier(
    estimators=[
        ('dt', best_dt),
        ('knn', best_knn),
        ('svc', best_svc),
        ('rf', best_rf),
        ('nb', best_nb),
        ('ridge', best_ridge),
        ('lr', best_lr),
        ('mlp1', best_mlp1),
        ('mlp2', best_mlp2)
    ],
    voting='hard'
)

print("\n🗳️  Evaluating Hard Voting Classifier...")
voting_score = cross_val_score(voting_clf, X_train, y_train, cv=cv, scoring="accuracy").mean()
print(f"Hard Voting Accuracy: {voting_score:.4f}")


# ---------- Stacking Ensemble ----------
stacking_clf = StackingClassifier(
    estimators=[
        ('dt', best_dt),
        ('knn', best_knn),
        ('svc', best_svc),
        ('rf', best_rf),
        ('nb', best_nb),
        ('ridge', best_ridge),
        ('mlp1', best_mlp1),
        ('mlp2', best_mlp2)
    ],
    final_estimator=LogisticRegression(max_iter=1000, random_state=42),
    cv=5
)

print("\n📚 Evaluating Stacking Classifier...")
stacking_score = cross_val_score(stacking_clf, X_train, y_train, cv=cv, scoring="accuracy").mean()
print(f"Stacking Accuracy: {stacking_score:.4f}")


# ---------- Train all models and get test predictions ----------
print("\n" + "=" * 60)
print("TRAINING MODELS ON FULL TRAINING SET")
print("=" * 60)

models_dict = {
    "Decision Tree": best_dt,
    "K-Nearest Neighbors": best_knn,
    "Support Vector Classifier": best_svc,
    "Random Forest": best_rf,
    "Naive Bayes": best_nb,
    "Ridge Classifier": best_ridge,
    "Logistic Regression": best_lr,
    "Deep Learning (MLP-1)": best_mlp1,
    "Deep Learning (MLP-2)": best_mlp2,
    "Hard Voting Ensemble": voting_clf,
    "Stacking Ensemble": stacking_clf
}

cv_scores = {
    "Decision Tree": studies['dt'].best_value,
    "K-Nearest Neighbors": studies['knn'].best_value,
    "Support Vector Classifier": studies['svc'].best_value,
    "Random Forest": studies['rf'].best_value,
    "Naive Bayes": studies['nb'].best_value,
    "Ridge Classifier": studies['ridge'].best_value,
    "Logistic Regression": studies['lr'].best_value,
    "Deep Learning (MLP-1)": studies['mlp1'].best_value,
    "Deep Learning (MLP-2)": studies['mlp2'].best_value,
    "Hard Voting Ensemble": voting_score,
    "Stacking Ensemble": stacking_score
}

# Dictionary to store predictions
predictions_dict = {}

print("\nTraining and predicting...")
for name, model in models_dict.items():
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    predictions_dict[name] = y_pred
    test_acc = (y_pred == y_test).mean()
    print(f"✓ {name}: Test Accuracy = {test_acc:.4f}")


# ---------- Final Comparison ----------
print("\n" + "=" * 60)
print("📊 FINAL RESULTS SUMMARY")
print("=" * 60)

results = [(name, score) for name, score in cv_scores.items()]
results.sort(key=lambda x: x[1], reverse=True)

print("\nRanked by Cross-Validation Accuracy:")
for i, (name, acc) in enumerate(results, 1):
    marker = "🏆" if i == 1 else "🥈" if i == 2 else "🥉" if i == 3 else "  "
    print(f"{marker} {i}. {name:.<40} {acc:.4f}")

# ---------- Save best model predictions ----------
best_model_name = results[0][0]
best_cv_score = results[0][1]

# Get predictions from best model
best_y_pred = predictions_dict[best_model_name]
best_model = models_dict[best_model_name]

print("\n" + "=" * 60)
print(f"🎉 Best Model: {best_model_name}")
print(f"   CV Accuracy: {best_cv_score:.4f}")
print(f"   Test Accuracy: {(best_y_pred == y_test).mean():.4f}")
print("=" * 60)

print("\n" + "=" * 60)
print("SAVED VARIABLES FOR FURTHER EVALUATION")
print("=" * 60)
print("""
Available variables:
  • y_test                - True labels for test set
  • best_y_pred           - Predictions from best model
  • best_model            - Trained best model object
  • best_model_name       - Name of best model
  • predictions_dict      - All predictions: predictions_dict['Model Name']
  • models_dict           - All trained models: models_dict['Model Name']
  • X_train, X_test       - Feature sets
  • y_train, y_test       - Label sets

Example usage:
  from sklearn.metrics import classification_report, confusion_matrix
  print(classification_report(y_test, best_y_pred))
  print(confusion_matrix(y_test, best_y_pred))
  
  # Compare any two models:
  print(confusion_matrix(y_test, predictions_dict['Random Forest']))
""")