import numpy as np
import pandas as pd
import optuna
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.metrics import *
from sklearn.base import BaseEstimator
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC


class BinaryClassifierNCV(BaseEstimator):
    """Class to tune the hyperparameters of a classifier using optuna and nested cross validation,
    the specified inner cross validation folds are used for hyperparameter tuning and the specified outer folds are used to evaluate the model performance.
    """

    def __init__(self, classifier_name, seed=42, outer_folds=5, inner_folds=3, shuffle=True, trial_num=10, verbose=True, model=None):
        self.shuffle = shuffle
        self.outer_folds = outer_folds
        self.inner_folds = inner_folds
        self.trial_num = trial_num
        self.classifier_name = classifier_name
        self.model = model
        self.shuffle
        self.verbose = verbose
        # Set the random seed for generating the seed for each trial
        np.random.seed(seed)
        self.trials_seeds = np.random.randint(0, 1000, size=trial_num)
        if self.verbose:
            print("The random seeds for each trial are:",
                  self.trials_seeds, "\n")

    # Fit the selected classifier in 10 cross validation trials for hyperparameter tuning
    def fit(self, X, y):
        # Initializing Naive Bayes classifier to be used as a baseline model
        baseline_model = GaussianNB()
        # Initialize an array to store the best parameters for outer cross validation loop
        parameters = []
        # Initializing the dataframes to store the average scores per trial
        scores_df = pd.DataFrame(columns=["F1", "Balanced Accuracy", "Precision", "MCC", "Recall", "ROC AUC"], index=[
                                 f"Trial {i + 1}" for i in range(self.trial_num)])
        baseline_df = pd.DataFrame(columns=["F1", "Balanced Accuracy", "Precision", "MCC", "Recall", "ROC AUC"], index=[
                                   f"Trial {i + 1}" for i in range(self.trial_num)])

        ############################# Nested cross validation for hyperparameter tuning ###################################################
        # objective function to be optimized by optuna for each inner cross validation loop
        # This function is called inside the for loop of the outer cross validation loop
        def objective(trial):
            # Knearst neighbors tuning
            if self.classifier_name == "KNeighborsclassifier":
                n_neighbors = trial.suggest_int("n_neighbors", 2, 10)
                weights = trial.suggest_categorical(
                    "weights", ["uniform", "distance"])
                p = trial.suggest_int("p", 1, 2)
                self.model = KNeighborsClassifier(
                    n_neighbors=n_neighbors, weights=weights, p=p)
            # LDA tuning
            if self.classifier_name == "LinearDiscriminantAnalysis":
                solver = trial.suggest_categorical(
                    "solver", ["svd", "lsqr", "eigen"])
                store_covariance = trial.suggest_categorical(
                    "store_covariance", [True, False])
                self.model = LinearDiscriminantAnalysis(
                    solver=solver, store_covariance=store_covariance)
            # Logistic regression tuning
            if self.classifier_name == "LogisticRegression":
                C = trial.suggest_float("C", 0, 10)
                penalty = trial.suggest_categorical("penalty", ["l1", "l2"])
                self.model = LogisticRegression(
                    C=C, penalty=penalty, solver='liblinear')
            # SVM tuning
            if self.classifier_name == "SVC":
                kernel = trial.suggest_categorical(
                    "kernel", ["linear", "poly", "rbf", "sigmoid"])
                C = trial.suggest_float("C", 0, 10)
                self.model = SVC(C=C, kernel=kernel)
            # Fit the  model and predict on the outer test sets after hyperparameter tuning in the inner cross validation loop
            cost_function = make_scorer(cohen_kappa_score)
            scores = cross_val_score(
                self.model, X_train, y_train, cv=self.inner_cv, scoring=cost_function)
            return scores.mean()

        ##################### Outer cross validation loop to evade overfitting on a single train/test split ############################
        # Number of of outer cross validation loops deterimined by the number of seeds for the trials
        trial_counter = 0
        for seed in self.trials_seeds:
            trial_scores = np.zeros((self.outer_folds, 6))
            base_trial_scores = np.zeros((self.outer_folds, 6))
            outer_cv_counter = 0
            # Create the stratified fold objects for the outer and inner cross validation loops with different random states per trial
            self.outer_cv = StratifiedKFold(
                n_splits=self.outer_folds, shuffle=self.shuffle, random_state=seed)
            self.inner_cv = StratifiedKFold(
                n_splits=self.inner_folds, shuffle=self.shuffle, random_state=seed)

            for train_index, test_index in self.outer_cv.split(X, y):
                # Outer cross validation loop to evade overfitting on a single train/test split
                X_train, X_test = X[train_index], X[test_index]
                y_train, y_test = y[train_index], y[test_index]
                # inner cross validation loop to tune hyperparameterss
                study = optuna.create_study(direction="maximize")
                study.optimize(objective, n_trials=50)
                params = study.best_params

                # Set the best hyperparameters for the model
                self.model.set_params(**params)
                parameters.append(params)

                # Fit the model and predict on the outer test sets after hyperparameter tuning in the inner cross validation loop
                self.model.fit(X_train, y_train)
                y_pred = self.model.predict(X_test)
                # Fit baseline classifier_name and predict on the outer test sets
                baseline_model.fit(X_train, y_train)
                y_base_pred = baseline_model.predict(X_test)

                # Calculate scores for each outer cross validation loop and trial
                trial_scores[outer_cv_counter, :] = self.score(y_test, y_pred)
                base_trial_scores[outer_cv_counter,
                                  :] = self.score(y_test, y_base_pred)
                # Move to the next outer cross validation fold
                outer_cv_counter += 1
            # Increase the trial counter and go to the next trial
            scores_df.loc[f"Trial {trial_counter + 1}"] = trial_scores.mean(
                axis=0)
            baseline_df.loc[f"Trial {trial_counter + 1}"] = base_trial_scores.mean(
                axis=0)
            trial_counter += 1

        ######################### Results section ########################################
        if self.verbose:
            mean_var_df = pd.concat([scores_df.mean(axis=0), scores_df.var(
                axis=0)], axis=1).rename(columns={0: "Mean", 1: "Variance"})
            base_mean_var_df = pd.concat([baseline_df.mean(axis=0), baseline_df.var(
                axis=0)], axis=1).rename(columns={0: "Mean", 1: "Variance"})
            print(
                f"The average scores and their variance for {self.classifier_name} across all {self.trial_num} trials are:")
            print(mean_var_df, "\n")
            print(mean_var_df.mean(axis=0))
            print(
                f"The average Naive Bayes baseline scores and their variance across all {self.trial_num} trial are:")
            print(base_mean_var_df, "\n")
            print(base_mean_var_df.mean(axis=0))
        return scores_df, baseline_df, parameters

    def score(self, y_test, y_pred):
        return roc_auc_score(y_test, y_pred), matthews_corrcoef(y_test, y_pred), balanced_accuracy_score(y_test, y_pred),\
            f1_score(y_test, y_pred), precision_score(
                y_test, y_pred), recall_score(y_test, y_pred)
