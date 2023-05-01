import pandas as pd
import numpy as np
import random
import warnings
from tqdm import tqdm
from sklearn.metrics import f1_score, confusion_matrix
from sklearn.model_selection import StratifiedKFold
from sklearn.ensemble import (
    AdaBoostClassifier,
    RandomForestClassifier,
)
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import (
    BernoulliNB as BayesianClassifer,
    GaussianNB,
)
from sklearn.neural_network import MLPClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC

from utils.data_process import data_processing
from utils.dataset_loader import load_dataset
from utils.feature_select import feature_selecting

from utils.dataset_loader import *

#Ignore warnings
warnings.filterwarnings("ignore")

def model_training(classifier_name, feature_selection_name):

    # List of dataset filenames
    dataset_files = load_dataset()

    confusion_matrices = []
    accuracies = []
    f_measures = []
    
    for dataset_file in tqdm(dataset_files, desc="Dataset Progress"):
        # Load dataset
        data = pd.read_excel(f'D:\\uit\\BaoMatWeb\\MLDroid\\DATASET\\{dataset_file}')

        #Process data
        y = data['Class']
        X_new = data_processing(data)

        #Perform feature selection
        X_new = feature_selecting(feature_selection_name, X_new, y)

        if classifier_name == "SVM":
            # Train a SVM classifier using 20-fold cross-validation
            clf = SVC(kernel='poly', C=69, random_state=None, max_iter=1000)
            pass
        elif classifier_name == "NB":
            # Train a Naive Bayes classifier using 20-fold cross-validation
            clf = GaussianNB(var_smoothing=random.uniform(1e-9, 1e-6))
            pass
        elif classifier_name == "RF":
            # Train a Random Forest classifier using 20-fold cross-validation
            clf = RandomForestClassifier(n_estimators=100, random_state=None)
            pass
        elif classifier_name == "MLP":
            # Train a MLP classifier using 20-fold cross-validation
            clf = MLPClassifier(hidden_layer_sizes=(100, 50), max_iter=1000, random_state=None)
            pass
        elif classifier_name == "LR":
            # Train a Logistic Regression classifier using 20-fold cross-validation
            clf = clf = LogisticRegression(random_state=None)
            pass
        elif classifier_name == "BN":
            # Train a Bayesian Network classifier using 20-fold cross-validation
            clf = BayesianClassifer()
            pass
        elif classifier_name == "AB":
            # Train an AdaBoost classifier using 20-fold cross-validation
            clf = AdaBoostClassifier(n_estimators=100, random_state=None)
            pass
        elif classifier_name == "DT":
            # Train a Decision Tree classifier using 20-fold cross-validation
            clf = DecisionTreeClassifier(random_state=None)
            pass
        elif classifier_name == "KNN":
            # Train a k-Nearest Neighbors classifier using 20-fold cross-validation
            clf = KNeighborsClassifier(weights=random.choice(['uniform', 'distance']))
            pass
        elif classifier_name == "DNN":
            # Train a Deep Neural Network classifier using 20-fold cross-validation
            clf = MLPClassifier(hidden_layer_sizes=(200, 150, 100, 50), max_iter=1000)
            pass
        else:
            print("Invalid classifier name.")

        sk_folds = StratifiedKFold(n_splits=20, shuffle=True, random_state=None)

        for train_index, test_index in sk_folds.split(X_new, y):
            X_train, X_test = X_new[train_index], X_new[test_index]
            y_train, y_test = y.iloc[train_index], y.iloc[test_index]

            clf.fit(X_train, y_train)
            y_pred = clf.predict(X_test)

            confusion_matrices.append(confusion_matrix(y_test, y_pred))

            accuracy = clf.score(X_test, y_test)
            accuracies.append(accuracy)

            f_measure = f1_score(y_test, y_pred, average='weighted')
            f_measures.append(f_measure)

        accuracy_mean = np.mean(accuracies)
        f_measure_mean = np.mean(f_measures)

        # Print mean accuracy and f1 score
        print(f"{dataset_file}: Accuracy: {accuracy_mean*100:.2f}  F-measure: {f_measure_mean:.2f}")
