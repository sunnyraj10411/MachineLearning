import sys
import numpy as np
from sklearn.model_selection import GridSearchCV
from sklearn.kernel_ridge import KernelRidge

'''
Usage:
python KRR_sklearn.py ..\dataSets\abalone\rings\Prototask.data\data
'''


def read_data(dataPath):
    import pandas as pd
    CONVERTER = lambda x: {'M':1,'F':2,'I':3}[x]
    CSV_COLUMNS = ["sex","length","diameter","height","whole_weight","shucked_weight","viscera_weight","shell_weight","rings"]
    frame = pd.read_table(dataPath,encoding='utf-8',sep=' ', index_col=None,header=None,names=CSV_COLUMNS,converters={0:CONVERTER})
    return frame

def get_features_and_labels(frame,test_percentage = 0.3):
    # Convert values to floats
    arr = np.array(frame, dtype=np.float)

    # Normalize the entire data set
    #from sklearn.preprocessing import StandardScaler, MinMaxScaler
    #arr = MinMaxScaler().fit_transform(arr)

    # Use the last column as the target value
    X, y = arr[:, :-1], arr[:, -1]

    # Use 80% of the data for training`
    from sklearn.model_selection import train_test_split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_percentage,random_state=0)

    # Normalize the attribute values to mean=0 and variance=1
    #from sklearn.preprocessing import StandardScaler
    #scaler = StandardScaler()

    ### Fit the scaler based on the training data, then apply the same
    ### scaling to both training and test sets.
    #scaler.fit(X_train)
    #X_train = scaler.transform(X_train)
    #X_test = scaler.transform(X_test)

    # Return the training and test sets
    return X_train, X_test, y_train, y_test

def std_func(y_true, y_pred):
    return np.power(np.sum(np.power(y_pred-y_true,2))/len(y_true),0.5)

def get_best_model(X_train, X_test, y_train, y_test):
    from sklearn.metrics import make_scorer
    clf = GridSearchCV(KernelRidge(), tuned_parameters, cv=5, scoring=make_scorer(std_func,greater_is_better=False))
    clf.fit(X_train, y_train)

    print("Best parameters set found on training set:")
    print()
    print(clf.best_params_)
    print()
    print("Grid scores on training set:")
    print()
    means = clf.cv_results_['mean_test_score']
    stds = clf.cv_results_['std_test_score']
    for mean, std, params in zip(means, stds, clf.cv_results_['params']):
        print("%0.3f (+/-%0.03f) for %r"
              % (mean, std * 2, params))
    print()
    print("The standard deviation of applying the best model on testing set:")
    y_true, y_pred = y_test, clf.predict(X_test)
    print(std_func(y_test, y_pred))
    print()



if len(sys.argv) != 2:
    print("Usage: ",sys.argv[0],"dataFile")
    exit()
frame = read_data(sys.argv[1])
X_train, X_test, y_train, y_test = get_features_and_labels(frame)
alpha_ridge = [1e-10,1e-5,1,2,5,10]
tuned_parameters = [{'kernel': ['rbf'], 'alpha': alpha_ridge},
                    {'kernel': ['linear'], 'alpha': alpha_ridge},
                    {'kernel': ['poly'], 'alpha': alpha_ridge}]


get_best_model(X_train, X_test, y_train, y_test)
