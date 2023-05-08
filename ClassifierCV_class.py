import numpy as np
import pandas as pd
import optuna
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.metrics import *
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.utils import compute_sample_weight
from sklearn.utils.estimator_checks import check_estimator


class ClassifierCV():
    """Class to provide a robust evaluation of classifiers as well as hyperparameter tuning using optuna 
    for binary classification problems.
    1. Use fit_tranform method to rigorously evalaute different classifiers with automatic hyperparameter tuning.
    2. Use fit method to tune and fit the best classifier on the entire dataset.
    3. the maximization of the mean F2 (fbeta) score across the tuning cv folds is used to tune the hyperparameters,
    thus the use can choose to place more importance on precision or recall based on the classification problem
    The specified tuning cross validation folds are used for hyperparameter tuning and the specified 
    outer folds are used to evaluate the model performance in fit_evaluate. This allows for a robust evalution of the
    selected classifier that is not biased by a single train/test split"""

    def __init__(self, classifier_name, class_weight={1: 1, 0: 1}, tuning_beta=1, model=None, seed=42, outer_folds=5, tuning_folds=3, optimization_trials=50, trial_num=10, shuffle=True, verbose=True):
        """
        classifier_name: Name of the classifier to be used for hyperparameter tuning (e.g. "KNeighborsClassifier")
        tuning_beta: fbeta score in the tuning cv loop, choose beta < 1 to prioritize precision, beta > 1 to prioritize recall (Less FN) (default=1 == f1 score)
        model: This is set to None, it will be used to store the best model after hyperparameter tuning
        class_weights: dictionary in the form {class_label_1: weight_1,class_label_2,weight_2} to adjust the importance of classes in the training process default(equal importance {1: 1, 0: 1})
        seed: Random seed for generating the seed array from which a seed is picked for each trial (default=42) 
        outer_folds: Number of outer cross validation folds (default=5)
        tuning_folds: Number of tuning cross validation folds, used in `fit_evaluate` as nested cv or in fit as simple cv (default=3) 
        shuffle: Boolean to shuffle the data before splitting into folds (default=True)
        optimization_trials: Number of trials optuna will perform to optimize the objective function (default=50)
        trial_num: Number of trials to be performed for hyperparameter tuning (default=10)
        verbose: Boolean, if Ture print the results of the fit method after the trials finish (default=True)
        """
        self.class_weight = class_weight
        self.shuffle = shuffle
        self.outer_folds = outer_folds
        self.tuning_folds = tuning_folds
        self.trial_num = trial_num
        self.classifier_name = classifier_name
        self.tuning_beta = tuning_beta
        self.model = model
        self.shuffle
        self.optimization_trials = optimization_trials
        self.verbose = verbose
        # Set the random seed for generating the seed for each trial
        self.seed = seed
        np.random.seed(seed)

    def __repr__(self):
        """Class Object string representation"""
        return "ClassifierCV()"

    def get_params(self, deep=False):
        """method to get the parameters of an estimator, required for implementation in sklearn pipelines"""
        param_dict = {"classifier_name": self.classifier_name, "class_weight": self.class_weight, "tuning_beta": self.tuning_beta,
                      "model": self.model, "seed": self.seed, "outer_folds": self.outer_folds,
                      "tuning_folds": self.tuning_folds, "optimization_trials": self.optimization_trials,
                      "trial_num": self.trial_num, "shuffle": self.shuffle, "verbose": self.verbose}
        return param_dict

    def set_params(self, **kwargs):
        """method to set the parameters of an estimator, required for implementation in sklearn pipelines"""
        # Update the parameters using keyword arguments
        for key, value in kwargs.items():
            setattr(self, key, value)
        return self

    # Fit the selected classifier in 10 cross validation trials for hyperparameter tuning
    def fit_transform(self, X, y):
        self.trials_seeds = np.random.randint(0, 1000, size=self.trial_num)
        """Nested cross validation with hyperparameter optimization in the inner cv loop to provide a robust evaluation of the model performance.
        Do not use this method to fit the final model, it is meant to be used for classifier category evaluation only"""
        # Initialize an array to store the best parameters for outer cross validation loop
        parameters = []
        # Initializing the dataframes to store the scores
        total_cv_scores_df = pd.DataFrame(columns=[
                                          "F1", "Balanced Accuracy", "Precision", "MCC", "Recall", "ROC AUC", "F2", "Specificity", "Negative PredVal"])
        if self.verbose:
            print("The seeds for each trial are: ", self.trials_seeds, "\n")
        # Number of of outer cross validation loops deterimined by the number of seeds for the trials
        for seed in self.trials_seeds:
            trial_scores = np.zeros((self.outer_folds, 9))
            outer_cv_counter = 0
            # Create the stratified fold objects for the outer and nested tuning cross validation loops with different random states per trial
            self.outer_cv = StratifiedKFold(
                n_splits=self.outer_folds, shuffle=self.shuffle, random_state=seed)
            self.tuning_cv = StratifiedKFold(
                n_splits=self.tuning_folds, shuffle=self.shuffle, random_state=seed)

            for train_index, test_index in self.outer_cv.split(X, y):
                ################# START Outer cross validation loop to evade overfitting on a single train/test split START ####################
                self.X_train, X_test = X[train_index], X[test_index]
                self.y_train, y_test = y[train_index], y[test_index]
                ############## START tuning cross validation loop to tune hyperparameters via the optuna objective method START ##################
                # Gaussian Naive Bayes does not have any hyperparameters to tune, skip the tuning cross validation loop
                if self.classifier_name != "GaussianNB":
                    # Make the sampler behave in a deterministic way.
                    sampler = optuna.samplers.TPESampler(seed=seed)
                    study = optuna.create_study(
                        direction="maximize", sampler=sampler)
                    study.optimize(
                        self.objective, n_trials=self.optimization_trials)
                    params = study.best_params
                    # Set the best hyperparameters for the model
                    self.model.set_params(**params)
                    parameters.append(params)
                else:
                    self.model = GaussianNB()
                ################### END tuning cross validation loop to tune hyperparameters via the optuna objective method END #################
                # Fit the tuned model and predict on the outer test sets after hyperparameter tuning in the tuning cross validation loop
                self.model.fit(self.X_train, self.y_train)
                y_pred = self.model.predict(X_test)
                # Calculate scores for each outer cross validation loop and trial
                trial_scores[outer_cv_counter,
                             :] = list(self.score(y_test, y_pred).values())
                # Move to the next outer cross validation fold
                outer_cv_counter += 1
            # Append by row the fold scores of this cross validation loop to the total_cv_scores_df
            temp_df = pd.DataFrame(
                trial_scores, columns=total_cv_scores_df.columns)
            total_cv_scores_df = pd.concat(
                [total_cv_scores_df, temp_df], axis=0, ignore_index=True)
        ################ END Outer cross validation loop to evade overfitting on a single train/test split END ######################
        ######################### Results section ########################################
        if self.verbose:
            mean_var_df = pd.concat([total_cv_scores_df.mean(axis=0), total_cv_scores_df.std(
                axis=0)], axis=1).rename(columns={0: "Mean", 1: "Stdev"})
            print(
                f"The average scores and their standard deviation for {self.classifier_name} across all {self.trial_num} trials are:")
            print(mean_var_df, "\n")
            print(mean_var_df.mean(axis=0))
        return total_cv_scores_df, parameters

    def fit(self, X, y, seed):
        """Simple Cross validation with hyperparameter optimization to provide a final optimized model,
        after the best category of classifier is selected via fit_evaluate. Use this method to 
        fit the final model in the whole dataset."""
        self.tuning_cv = StratifiedKFold(
            n_splits=self.tuning_folds, shuffle=self.shuffle, random_state=self.seed)
        self.X_train = X
        self.y_train = y
        sampler = optuna.samplers.TPESampler(seed=seed)
        study = optuna.create_study(direction="maximize", sampler=sampler)
        study.optimize(self.objective, n_trials=self.optimization_trials)
        params = study.best_params
        self.model.set_params(**params)
        self.model.fit(self.X_train, self.y_train)
        return self.model

    def predict(self, X):
        """Make prediction with the final optimized model"""
        return self.model.predict(X)

    def score(self, y_test, y_pred):
        """Method to to calculate different scores for the model performance"""
        # Compute the confusion matrix
        cm = confusion_matrix(y_test, y_pred)
        # Compute the scores
        roc = roc_auc_score(y_test, y_pred)
        mcc = matthews_corrcoef(y_test, y_pred)
        precision = precision_score(y_test, y_pred)
        balanced_accuracy = balanced_accuracy_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred)
        recall = recall_score(y_test, y_pred)
        f2 = fbeta_score(y_test, y_pred, beta=self.tuning_beta)
        specificity = cm[0, 0]/(cm[0, 0] + cm[0, 1])
        negative_pv = cm[0, 0]/(cm[0, 0] + cm[1, 0])
        # store them in a dictionary
        scores_dict = {"F1": f1, "Balanced Accuracy": balanced_accuracy, "Precision": precision, "MCC": mcc,
                       "Recall": recall, "ROC AUC": roc, "F2": f2, "Specificity": specificity, "Negative PredVal": negative_pv}
        return scores_dict
    ############################# Cross validation for hyperparameter tuning ###################################################
    # This method is called inside the for loop for each of the outer cross validations

    def objective(self, trial):
        """Objective function to be optimized by optuna for each tuning cross validation loop,
        It utilizes the classifier_name attribute to determine which classifier to optimize"""
        # Knearst neighbors tuning
        if self.classifier_name == "KNeighborsclassifier":
            n_neighbors = trial.suggest_int("n_neighbors", 2, 10)
            weights = trial.suggest_categorical(
                "weights", ["uniform", "distance"])
            p = trial.suggest_int("p", 1, 2)
            self.model = KNeighborsClassifier(
                n_neighbors=n_neighbors, weights=weights, p=p)
        # LDA tuning
        elif self.classifier_name == "LinearDiscriminantAnalysis":
            solver = trial.suggest_categorical(
                "solver", ["svd", "lsqr", "eigen"])
            store_covariance = trial.suggest_categorical(
                "store_covariance", [True, False])
            self.model = LinearDiscriminantAnalysis(
                solver=solver, store_covariance=store_covariance)
        # Lasso Logistic regression tuning
        elif self.classifier_name == "LogisticRegression-lasso":
            C = trial.suggest_float("C", 0.001, 100.0, log=True)
            penalty = "l1"
            solver = trial.suggest_categorical("solver", ['liblinear', 'saga'])
            self.model = LogisticRegression(
                C=C, penalty=penalty, solver=solver, class_weight=self.class_weight)
        # Elastic Logistic regression tuning
        elif self.classifier_name == "LogisticRegression-elastic":
            solver = 'saga'
            penalty = "elasticnet"
            C = trial.suggest_float("C", 0.001, 100.0, log=True)
            l1_ratio = trial.suggest_float("l1_ratio", 0.0, 0.1)
            self.model = LogisticRegression(
                C=C, solver=solver, penalty=penalty, l1_ratio=l1_ratio, class_weight=self.class_weight)
        # Ridge Logistic regression tuning
        elif self.classifier_name == "LogisticRegression-ridge":
            solver = trial.suggest_categorical(
                "solver", ['lbfgs', 'liblinear', 'newton-cg', 'sag', 'saga'])
            penalty = "l2"
            C = trial.suggest_float("C", 0.001, 100.0, log=True)
            self.model = LogisticRegression(
                C=C, penalty=penalty, solver=solver, class_weight=self.class_weight)
        # linear kernel SVM tuning
        elif self.classifier_name == "SVC-linear":
            kernel = 'linear'
            C = trial.suggest_float("C", 0.001, 100.0, log=True)
            self.model = SVC(C=C, kernel=kernel,
                             class_weight=self.class_weight, probability=True)
        # Polynomial kernel SVM tuning
        elif self.classifier_name == "SVC-poly":
            kernel = "poly"
            degree = trial.suggest_int('degree', 3, 5)
            C = trial.suggest_float("C", 0.001, 100.0, log=True)
            gamma = trial.suggest_float("gamma", 0.001, 100.0, log=True)
            self.model = SVC(C=C, kernel=kernel, gamma=gamma,
                             degree=degree, class_weight=self.class_weight, probability=True)
        # Radial basis function kernel SVM tuning
        elif self.classifier_name == "SVC-rbf":
            kernel = "rbf"
            C = trial.suggest_float("C", 0.001, 100.0, log=True)
            gamma = trial.suggest_float("gamma", 0.001, 100.0, log=True)
            self.model = SVC(C=C, kernel=kernel, gamma=gamma,
                             class_weight=self.class_weight, probability=True,)
        # sigmoid kernel SVM tuning
        elif self.classifier_name == "SVC-sigmoid":
            kernel = "sigmoid"
            C = trial.suggest_float("C", 0.001, 100.0, log=True)
            gamma = trial.suggest_float("gamma", 0.001, 100.0, log=True)
            self.model = SVC(C=C, kernel=kernel, gamma=gamma,
                             class_weight=self.class_weight, probability=True)
        else:
            raise ValueError(
                f"{self.classifier_name} is not a valid classifier name or its not yet implemented")
        # Fit the  model and predict on the outer test sets after hyperparameter tuning in the tuning cross validation loop
        inner_scores = []
        for train_index, test_index in self.tuning_cv.split(self.X_train, self.y_train):
            ################# Outer cross validation loop to evade overfitting on a single train/test split START ####################
            X_inner_train, X_inner_test = self.X_train[train_index], self.X_train[test_index]
            y_inner_train, y_inner_test = self.y_train[train_index], self.y_train[test_index]
            self.model.fit(X_inner_train, y_inner_train)
            y_inner_pred = self.model.predict(X_inner_test)
            inner_scores.append(fbeta_score(
                y_inner_test, y_inner_pred, beta=self.tuning_beta))
        return np.array(inner_scores).mean()
