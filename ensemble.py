#take best hyperparameters for models. Make the best ensemble. Evaluate at the same time for choosing.
#choose how the models can be evaluated to be the "best"


"""
Comprehensive hyperparameter tuning for multiple classifiers
with ensemble methods (Hard Voting & Stacking)
Uses TensorFlow/Keras for deep learning models
"""

import optuna
import numpy as np
from sklearn.datasets import load_wine
from sklearn.model_selection import cross_val_score, StratifiedKFold, train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier, VotingClassifier, StackingClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import RidgeClassifier, LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.base import BaseEstimator, ClassifierMixin
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping
import warnings
warnings.filterwarnings('ignore')

# Load dataset
X, y = load_wine(return_X_y=True)

# Split data for final evaluation
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# Define CV strategy (on training data)
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

# Suppress optuna logging for cleaner output
optuna.logging.set_verbosity(optuna.logging.WARNING)

# Set random seeds for reproducibility
np.random.seed(42)
tf.random.set_seed(42)


# ---------- Keras Wrapper for sklearn compatibility ----------
class KerasClassifierWrapper(BaseEstimator, ClassifierMixin):
    """Wrapper to make Keras models compatible with sklearn"""
    
    def __init__(self, build_fn, epochs=100, batch_size=32, verbose=0):
        self.build_fn = build_fn
        self.epochs = epochs
        self.batch_size = batch_size
        self.verbose = verbose
        self.model = None
        self.scaler = StandardScaler()
        self.history_ = None
        self.classes_ = None
        
    def fit(self, X, y):
        self.classes_ = np.unique(y)
        n_classes = len(self.classes_)
        
        # Scale features
        X_scaled = self.scaler.fit_transform(X)
        
        # Convert labels to categorical
        y_cat = keras.utils.to_categorical(y, n_classes)
        
        # Build model
        self.model = self.build_fn()
        
        # Early stopping
        early_stop = EarlyStopping(monitor='val_loss', patience=15, restore_best_weights=True)
        
        # Train
        history = self.model.fit(
            X_scaled, y_cat,
            epochs=self.epochs,
            batch_size=self.batch_size,
            verbose=self.verbose,
            callbacks=[early_stop],
            validation_split=0.2
        )
        self.history_ = history
        
        return self
    
    def predict(self, X):
        X_scaled = self.scaler.transform(X)
        predictions = self.model.predict(X_scaled, verbose=0)
        return np.argmax(predictions, axis=1)
    
    def predict_proba(self, X):
        X_scaled = self.scaler.transform(X)
        return self.model.predict(X_scaled, verbose=0)


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


# ---------- 8️⃣ Deep Learning (TensorFlow/Keras Sequential) ----------
def objective_dl(trial):
    # Architecture parameters
    n_layers = trial.suggest_int("n_layers", 1, 3)
    units_layer1 = trial.suggest_categorical("units_layer1", [32, 64, 128, 256])
    units_layer2 = trial.suggest_categorical("units_layer2", [32, 64, 128]) if n_layers >= 2 else 0
    units_layer3 = trial.suggest_categorical("units_layer3", [32, 64]) if n_layers >= 3 else 0
    dropout_rate = trial.suggest_float("dropout_rate", 0.0, 0.5)
    
    # Training parameters
    learning_rate = trial.suggest_float("learning_rate", 1e-4, 1e-2, log=True)
    batch_size = trial.suggest_categorical("batch_size", [16, 32])
    
    def build_model():
        model = Sequential()
        model.add(Dense(units_layer1, activation='relu', input_shape=(13,)))
        if dropout_rate > 0:
            model.add(Dropout(dropout_rate))
        
        if n_layers >= 2:
            model.add(Dense(units_layer2, activation='relu'))
            if dropout_rate > 0:
                model.add(Dropout(dropout_rate))
        
        if n_layers >= 3:
            model.add(Dense(units_layer3, activation='relu'))
            if dropout_rate > 0:
                model.add(Dropout(dropout_rate))
        
        model.add(Dense(3, activation='softmax'))
        
        model.compile(
            optimizer=keras.optimizers.Adam(learning_rate=learning_rate),
            loss='categorical_crossentropy',
            metrics=['accuracy']
        )
        return model
    
    # Create wrapper
    keras_clf = KerasClassifierWrapper(
        build_fn=build_model,
        epochs=100,
        batch_size=batch_size,
        verbose=0
    )
    
    # Evaluate with cross-validation
    score = cross_val_score(keras_clf, X_train, y_train, cv=cv, scoring="accuracy").mean()
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

print("\n🧠 Tuning Deep Learning Models (TensorFlow/Keras)...")
studies['dl'] = optuna.create_study(direction="maximize")
studies['dl'].optimize(objective_dl, n_trials=50, show_progress_bar=True)
print(f"Best accuracy: {studies['dl'].best_value:.4f}")

# Get top 2 DL configurations
all_trials = sorted(studies['dl'].trials, key=lambda t: t.value if t.value is not None else -1, reverse=True)
best_dl_params_1 = all_trials[0].params
best_dl_params_2 = all_trials[1].params
print(f"Second best accuracy: {all_trials[1].value:.4f}")


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

# Build DL models from best parameters
def build_dl_model_1():
    params = best_dl_params_1
    model = Sequential()
    model.add(Dense(params['units_layer1'], activation='relu', input_shape=(13,)))
    if params['dropout_rate'] > 0:
        model.add(Dropout(params['dropout_rate']))
    
    if params['n_layers'] >= 2:
        model.add(Dense(params['units_layer2'], activation='relu'))
        if params['dropout_rate'] > 0:
            model.add(Dropout(params['dropout_rate']))
    
    if params['n_layers'] >= 3:
        model.add(Dense(params['units_layer3'], activation='relu'))
        if params['dropout_rate'] > 0:
            model.add(Dropout(params['dropout_rate']))
    
    model.add(Dense(3, activation='softmax'))
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=params['learning_rate']),
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )
    return model

def build_dl_model_2():
    params = best_dl_params_2
    model = Sequential()
    model.add(Dense(params['units_layer1'], activation='relu', input_shape=(13,)))
    if params['dropout_rate'] > 0:
        model.add(Dropout(params['dropout_rate']))
    
    if params['n_layers'] >= 2:
        model.add(Dense(params['units_layer2'], activation='relu'))
        if params['dropout_rate'] > 0:
            model.add(Dropout(params['dropout_rate']))
    
    if params['n_layers'] >= 3:
        model.add(Dense(params['units_layer3'], activation='relu'))
        if params['dropout_rate'] > 0:
            model.add(Dropout(params['dropout_rate']))
    
    model.add(Dense(3, activation='softmax'))
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=params['learning_rate']),
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )
    return model

best_dl1 = KerasClassifierWrapper(
    build_fn=build_dl_model_1,
    epochs=100,
    batch_size=best_dl_params_1['batch_size'],
    verbose=0
)

best_dl2 = KerasClassifierWrapper(
    build_fn=build_dl_model_2,
    epochs=100,
    batch_size=best_dl_params_2['batch_size'],
    verbose=0
)


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
        ('dl1', best_dl1),
        ('dl2', best_dl2)
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
        ('dl1', best_dl1),
        ('dl2', best_dl2)
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
    "Deep Learning (DL-1)": best_dl1,
    "Deep Learning (DL-2)": best_dl2,
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
    "Deep Learning (DL-1)": all_trials[0].value,
    "Deep Learning (DL-2)": all_trials[1].value,
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

# ---------- Save best model predictions and histories ----------
best_model_name = results[0][0]
second_best_model_name = results[1][0]
third_best_model_name = results[2][0]

# Get predictions from top 3 models
best_y_pred = predictions_dict[best_model_name]
second_best_y_pred = predictions_dict[second_best_model_name]
third_best_y_pred = predictions_dict[third_best_model_name]

best_model = models_dict[best_model_name]
second_best_model = models_dict[second_best_model_name]
third_best_model = models_dict[third_best_model_name]

# Extract history for models that have it (only DL models)
best_history = best_model.history_ if hasattr(best_model, 'history_') else None
second_best_history = second_best_model.history_ if hasattr(second_best_model, 'history_') else None
third_best_history = third_best_model.history_ if hasattr(third_best_model, 'history_') else None

print("\n" + "=" * 60)
print(f"🏆 #1 Best Model: {best_model_name}")
print(f"   CV Accuracy: {results[0][1]:.4f}")
print(f"   Test Accuracy: {(best_y_pred == y_test).mean():.4f}")
print(f"   Training History: {'Available' if best_history is not None else 'N/A (not a DL model)'}")

print(f"\n🥈 #2 Best Model: {second_best_model_name}")
print(f"   CV Accuracy: {results[1][1]:.4f}")
print(f"   Test Accuracy: {(second_best_y_pred == y_test).mean():.4f}")
print(f"   Training History: {'Available' if second_best_history is not None else 'N/A (not a DL model)'}")

print(f"\n🥉 #3 Best Model: {third_best_model_name}")
print(f"   CV Accuracy: {results[2][1]:.4f}")
print(f"   Test Accuracy: {(third_best_y_pred == y_test).mean():.4f}")
print(f"   Training History: {'Available' if third_best_history is not None else 'N/A (not a DL model)'}")
print("=" * 60)

print("\n" + "=" * 60)
print("SAVED VARIABLES FOR FURTHER EVALUATION")
print("=" * 60)
print("""
Available variables for TOP 3 MODELS:
  • y_test                - True labels for test set
  
  • best_y_pred           - Predictions from #1 best model
  • second_best_y_pred    - Predictions from #2 best model
  • third_best_y_pred     - Predictions from #3 best model
  
  • best_model            - Trained #1 best model object
  • second_best_model     - Trained #2 best model object
  • third_best_model      - Trained #3 best model object
  
  • best_history          - Training history #1 (Keras History object or None)
  • second_best_history   - Training history #2 (Keras History object or None)
  • third_best_history    - Training history #3 (Keras History object or None)
  
  • best_model_name       - Name of #1 best model
  • second_best_model_name - Name of #2 best model
  • third_best_model_name  - Name of #3 best model

All models available:
  • predictions_dict      - All predictions: predictions_dict['Model Name']
  • models_dict           - All trained models: models_dict['Model Name']
  • X_train, X_test       - Feature sets
  • y_train, y_test       - Label sets

Example usage for comparison:
  from sklearn.metrics import classification_report, confusion_matrix
  import matplotlib.pyplot as plt
  
  # Evaluate top 3 models
  print("Best Model:")
  print(classification_report(y_test, best_y_pred))
  
  # Plot training history (if DL model)
  if best_history is not None:
      plt.plot(best_history.history['accuracy'], label='Training Accuracy')
      plt.plot(best_history.history['val_accuracy'], label='Validation Accuracy')
      plt.xlabel('Epoch')
      plt.ylabel('Accuracy')
      plt.legend()
      plt.show()
      
  # Access loss curves
  if best_history is not None:
      train_loss = best_history.history['loss']
      val_loss = best_history.history['val_loss']
""")



import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    mean_squared_error, mean_absolute_error, r2_score,
    accuracy_score, precision_recall_fscore_support, 
    confusion_matrix, classification_report, roc_curve, auc
)
from sklearn.preprocessing import label_binarize
import os
import json
from datetime import datetime

class ModelEvaluator:
    def __init__(self, output_dir='model_evaluation_results'):
        """Initialize evaluator with output directory"""
        self.output_dir = output_dir
        self.timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        self.results_dir = os.path.join(output_dir, f'evaluation_{self.timestamp}')
        os.makedirs(self.results_dir, exist_ok=True)
        
    def determine_task_type(self, y_true):
        """Determine if it's regression or classification"""
        unique_vals = np.unique(y_true)
        if len(unique_vals) <= 20 and np.all(y_true == y_true.astype(int)):
            return 'classification'
        return 'regression'
    
    def evaluate_regression(self, y_true, y_pred, model_name):
        """Evaluate regression model"""
        metrics = {
            'MSE': mean_squared_error(y_true, y_pred),
            'RMSE': np.sqrt(mean_squared_error(y_true, y_pred)),
            'MAE': mean_absolute_error(y_true, y_pred),
            'R2': r2_score(y_true, y_pred),
            'MAPE': np.mean(np.abs((y_true - y_pred) / (y_true + 1e-10))) * 100
        }
        
        # Plot predictions vs actual
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))
        
        # Scatter plot
        axes[0].scatter(y_true, y_pred, alpha=0.5, edgecolors='k', linewidth=0.5)
        axes[0].plot([y_true.min(), y_true.max()], 
                     [y_true.min(), y_true.max()], 
                     'r--', lw=2, label='Perfect Prediction')
        axes[0].set_xlabel('Actual Values', fontsize=12)
        axes[0].set_ylabel('Predicted Values', fontsize=12)
        axes[0].set_title(f'{model_name} - Predictions vs Actual', fontsize=14, fontweight='bold')
        axes[0].legend()
        axes[0].grid(True, alpha=0.3)
        
        # Residual plot
        residuals = y_true - y_pred
        axes[1].scatter(y_pred, residuals, alpha=0.5, edgecolors='k', linewidth=0.5)
        axes[1].axhline(y=0, color='r', linestyle='--', lw=2)
        axes[1].set_xlabel('Predicted Values', fontsize=12)
        axes[1].set_ylabel('Residuals', fontsize=12)
        axes[1].set_title(f'{model_name} - Residual Plot', fontsize=14, fontweight='bold')
        axes[1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(os.path.join(self.results_dir, f'{model_name}_regression_plots.png'), 
                   dpi=300, bbox_inches='tight')
        plt.close()
        
        return metrics
    
    def evaluate_classification(self, y_true, y_pred, model_name):
        """Evaluate classification model"""
        metrics = {
            'Accuracy': accuracy_score(y_true, y_pred)
        }
        
        # Get precision, recall, f1 for each class
        precision, recall, f1, support = precision_recall_fscore_support(
            y_true, y_pred, average=None, zero_division=0
        )
        
        # Weighted averages
        metrics['Precision (weighted)'] = precision_recall_fscore_support(
            y_true, y_pred, average='weighted', zero_division=0
        )[0]
        metrics['Recall (weighted)'] = precision_recall_fscore_support(
            y_true, y_pred, average='weighted', zero_division=0
        )[1]
        metrics['F1 (weighted)'] = precision_recall_fscore_support(
            y_true, y_pred, average='weighted', zero_division=0
        )[2]
        
        # Confusion matrix
        cm = confusion_matrix(y_true, y_pred)
        
        # Plot confusion matrix
        plt.figure(figsize=(10, 8))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                   cbar_kws={'label': 'Count'})
        plt.title(f'{model_name} - Confusion Matrix', fontsize=14, fontweight='bold')
        plt.ylabel('True Label', fontsize=12)
        plt.xlabel('Predicted Label', fontsize=12)
        plt.tight_layout()
        plt.savefig(os.path.join(self.results_dir, f'{model_name}_confusion_matrix.png'), 
                   dpi=300, bbox_inches='tight')
        plt.close()
        
        # Classification report
        report = classification_report(y_true, y_pred, zero_division=0)
        with open(os.path.join(self.results_dir, f'{model_name}_classification_report.txt'), 'w') as f:
            f.write(report)
        
        return metrics, cm
    
    def plot_training_history(self, history, model_name):
        """Plot training history for Sequential models"""
        if history is None:
            return
        
        metrics_to_plot = []
        for key in history.keys():
            if not key.startswith('val_'):
                metrics_to_plot.append(key)
        
        n_metrics = len(metrics_to_plot)
        if n_metrics == 0:
            return
        
        fig, axes = plt.subplots(1, n_metrics, figsize=(6*n_metrics, 5))
        if n_metrics == 1:
            axes = [axes]
        
        for idx, metric in enumerate(metrics_to_plot):
            axes[idx].plot(history[metric], label=f'Training {metric}', linewidth=2)
            
            val_metric = f'val_{metric}'
            if val_metric in history:
                axes[idx].plot(history[val_metric], label=f'Validation {metric}', linewidth=2)
            
            axes[idx].set_xlabel('Epoch', fontsize=12)
            axes[idx].set_ylabel(metric.upper(), fontsize=12)
            axes[idx].set_title(f'{model_name} - {metric.upper()} History', 
                               fontsize=14, fontweight='bold')
            axes[idx].legend()
            axes[idx].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(os.path.join(self.results_dir, f'{model_name}_training_history.png'), 
                   dpi=300, bbox_inches='tight')
        plt.close()
    
    def evaluate_model(self, y_true, y_pred, model_name, history=None):
        """Evaluate a single model"""
        print(f"\n{'='*60}")
        print(f"Evaluating: {model_name}")
        print(f"{'='*60}")
        
        task_type = self.determine_task_type(y_true)
        print(f"Task Type: {task_type.upper()}")
        
        results = {
            'model_name': model_name,
            'task_type': task_type
        }
        
        if task_type == 'regression':
            metrics = self.evaluate_regression(y_true, y_pred, model_name)
            results['metrics'] = metrics
            
            print("\nRegression Metrics:")
            for metric, value in metrics.items():
                print(f"  {metric}: {value:.6f}")
        
        else:  # classification
            metrics, cm = self.evaluate_classification(y_true, y_pred, model_name)
            results['metrics'] = metrics
            results['confusion_matrix'] = cm.tolist()
            
            print("\nClassification Metrics:")
            for metric, value in metrics.items():
                print(f"  {metric}: {value:.6f}")
        
        # Plot training history if available
        if history is not None:
            print("\nTraining history detected - generating plots...")
            self.plot_training_history(history, model_name)
            results['has_training_history'] = True
        else:
            results['has_training_history'] = False
        
        return results
    
    def compare_models(self, all_results):
        """Create comparison visualizations"""
        if not all_results:
            return
        
        task_type = all_results[0]['task_type']
        model_names = [r['model_name'] for r in all_results]
        
        # Get common metrics
        if task_type == 'regression':
            metrics_to_compare = ['RMSE', 'MAE', 'R2', 'MAPE']
        else:
            metrics_to_compare = ['Accuracy', 'Precision (weighted)', 
                                 'Recall (weighted)', 'F1 (weighted)']
        
        # Filter metrics that exist in all results
        available_metrics = set(all_results[0]['metrics'].keys())
        for result in all_results[1:]:
            available_metrics &= set(result['metrics'].keys())
        
        metrics_to_compare = [m for m in metrics_to_compare if m in available_metrics]
        
        if not metrics_to_compare:
            return
        
        # Create comparison plot
        n_metrics = len(metrics_to_compare)
        fig, axes = plt.subplots(1, n_metrics, figsize=(6*n_metrics, 5))
        if n_metrics == 1:
            axes = [axes]
        
        colors = ['#1f77b4', '#ff7f0e', '#2ca02c']
        
        for idx, metric in enumerate(metrics_to_compare):
            values = [r['metrics'][metric] for r in all_results]
            bars = axes[idx].bar(range(len(model_names)), values, color=colors[:len(model_names)])
            
            # Add value labels on bars
            for bar in bars:
                height = bar.get_height()
                axes[idx].text(bar.get_x() + bar.get_width()/2., height,
                             f'{height:.4f}',
                             ha='center', va='bottom', fontsize=10)
            
            axes[idx].set_xticks(range(len(model_names)))
            axes[idx].set_xticklabels(model_names, rotation=45, ha='right')
            axes[idx].set_ylabel(metric, fontsize=12)
            axes[idx].set_title(f'{metric} Comparison', fontsize=14, fontweight='bold')
            axes[idx].grid(True, alpha=0.3, axis='y')
        
        plt.tight_layout()
        plt.savefig(os.path.join(self.results_dir, 'model_comparison.png'), 
                   dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"\n{'='*60}")
        print("Model Comparison Summary")
        print(f"{'='*60}")
        
        # Find best model for each metric
        for metric in metrics_to_compare:
            values = [r['metrics'][metric] for r in all_results]
            
            # For R2 and Accuracy, higher is better
            if metric in ['R2', 'Accuracy', 'Precision (weighted)', 
                         'Recall (weighted)', 'F1 (weighted)']:
                best_idx = np.argmax(values)
            else:  # For error metrics, lower is better
                best_idx = np.argmin(values)
            
            print(f"\nBest {metric}: {model_names[best_idx]} ({values[best_idx]:.6f})")
    
    def save_results(self, all_results):
        """Save all results to JSON"""
        # Convert numpy types to Python types for JSON serialization
        def convert_to_json_serializable(obj):
            if isinstance(obj, np.integer):
                return int(obj)
            elif isinstance(obj, np.floating):
                return float(obj)
            elif isinstance(obj, np.ndarray):
                return obj.tolist()
            return obj
        
        json_results = []
        for result in all_results:
            json_result = {
                'model_name': result['model_name'],
                'task_type': result['task_type'],
                'has_training_history': result['has_training_history'],
                'metrics': {k: convert_to_json_serializable(v) 
                           for k, v in result['metrics'].items()}
            }
            if 'confusion_matrix' in result:
                json_result['confusion_matrix'] = result['confusion_matrix']
            json_results.append(json_result)
        
        with open(os.path.join(self.results_dir, 'evaluation_results.json'), 'w') as f:
            json.dump(json_results, f, indent=2)
        
        print(f"\n{'='*60}")
        print(f"Results saved to: {self.results_dir}")
        print(f"{'='*60}\n")


# Main evaluation function
def evaluate_top_models(y_test, best_y_pred, second_best_y_pred, third_best_y_pred,
                       best_history=None, second_best_history=None, third_best_history=None,
                       output_dir='model_evaluation_results'):
    """
    Evaluate top 3 models with their predictions and training histories
    
    Parameters:
    -----------
    y_test : array-like
        True test labels/values
    best_y_pred : array-like
        Predictions from best model
    second_best_y_pred : array-like
        Predictions from second best model
    third_best_y_pred : array-like
        Predictions from third best model
    best_history : dict, optional
        Training history from best model (if Sequential)
    second_best_history : dict, optional
        Training history from second best model (if Sequential)
    third_best_history : dict, optional
        Training history from third best model (if Sequential)
    output_dir : str, optional
        Directory to save results
    """
    
    evaluator = ModelEvaluator(output_dir=output_dir)
    
    # Evaluate each model
    all_results = []
    
    models_data = [
        ('Best Model', best_y_pred, best_history),
        ('Second Best Model', second_best_y_pred, second_best_history),
        ('Third Best Model', third_best_y_pred, third_best_history)
    ]
    
    for model_name, y_pred, history in models_data:
        result = evaluator.evaluate_model(y_test, y_pred, model_name, history)
        all_results.append(result)
    
    # Compare models
    evaluator.compare_models(all_results)
    
    # Save results
    evaluator.save_results(all_results)
    
    return all_results


# Usage example:
"""
# Get training histories if models are Sequential
best_history = best_model.history_ if hasattr(best_model, 'history_') else None
second_best_history = second_best_model.history_ if hasattr(second_best_model, 'history_') else None
third_best_history = third_best_model.history_ if hasattr(third_best_model, 'history_') else None

# Run evaluation
results = evaluate_top_models(
    y_test=y_test,
    best_y_pred=best_y_pred,
    second_best_y_pred=second_best_y_pred,
    third_best_y_pred=third_best_y_pred,
    best_history=best_history,
    second_best_history=second_best_history,
    third_best_history=third_best_history,
    output_dir='model_evaluation_results'
)
"""