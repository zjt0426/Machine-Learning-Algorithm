import numpy as np
import pandas as pd
from pandas import set_option
import seaborn as sns
from pandas.plotting import scatter_matrix
import matplotlib.pyplot as plt
from sklearn import datasets
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import Lasso
from sklearn.linear_model import ElasticNet
from sklearn.tree import DecisionTreeRegressor
from sklearn.svm import SVR
from sklearn.neighbors import KNeighborsRegressor
from sklearn.pipeline import Pipeline

class Linear:

    def __init__(self):
        pass
    def datatreating(self):


        # 数据理解
        data = datasets.load_boston()
        X = data.data     # 前13个数据
        Y = data.target  # 房价数据
        data = pd.DataFrame(X, columns=data.feature_names)
        data['PRICE'] = Y
        # print(data)
        # print(data.shape)
        # print(data.dtypes)
        # print(data.describe())
        # print(data.info())
        set_option('display.width', 120)
        # print(data.head(30))
        set_option('precision', 2)  # 小数
        # print(data.corr(method='pearson'))
        # data.hist()    # 直方图
        # plt.show()
        # data.plot(kind='density', subplots=True, layout=(4, 4), sharex=False, fontsize=1)    # 密度图
        # plt.show()
        # data.plot(kind='box', subplots=True, layout=(4, 4), sharex=False, sharey=False, fontsize=8)   # 箱型图
        # plt.show()
        # scatter_matrix(data)
        # plt.show()
        # plt.figure(figsize=(10, 8))
        # new_df = data.corr()
        # sns.heatmap(new_df, annot=True, vmax=1, square=True)
        # plt.savefig(r'D:\sim\heatmap.png')
        # plt.show()

        # 特征选择
        data = data.values
        X = data[:, 0:13]
        y = data[:, 13]
        seed = 7
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=seed)
        return X_train, X_test, y_train, y_test
    def modelsss(self):
        models = {}
        models['LR'] = LinearRegression()
        models['SVM'] = SVR()
        models['Tree'] = DecisionTreeRegressor()
        models['Lasso'] = Lasso()
        models['En'] = ElasticNet()
        models['KNN'] = KNeighborsRegressor()
        result = []

        for key in models:
            kfold = KFold(n_splits=10, shuffle=False)
            cv_result = cross_val_score(models[key], X_train, y_train, cv=kfold, scoring='neg_mean_squared_error')
            result.append(cv_result)

            print('%s: %f (%f)' % (key, cv_result.mean(), cv_result.std()))

        fig = plt.figure(figsize=(10, 8))
        fig.suptitle('Algorithm Comparison')
        ax = fig.add_subplot(111)
        plt.boxplot(result)
        ax.set_xticklabels(models.keys())
        plt.show()

    def normalize(self):
        # 正态化处理


        pipelines = {}
        pipelines['ScalerLR'] = Pipeline([('Scaler', StandardScaler()), ('LR', LinearRegression())])
        pipelines['ScalerLASSO'] = Pipeline([('Scaler', StandardScaler()), ('LASSO', Lasso())])
        pipelines['ScalerEN'] = Pipeline([('Scaler', StandardScaler()), ('EN', ElasticNet())])
        pipelines['ScalerKNN'] = Pipeline([('Scaler', StandardScaler()), ('KNN', KNeighborsRegressor())])
        pipelines['ScalerTREE'] = Pipeline([('Scaler', StandardScaler()), ('TREE', DecisionTreeRegressor())])
        pipelines['ScalerSVM'] = Pipeline([('Scaler', StandardScaler()), ('SVM', SVR())])

        results_ = []
        for key in pipelines:
            kfold = KFold(n_splits=10, shuffle=False)
            pip_result = cross_val_score(pipelines[key], X_train, y_train, cv=kfold, scoring='neg_mean_squared_error')
            results_.append(pip_result)
            print('%s : %f (%f)' % (key, pip_result.mean(), pip_result.std()))

        fig = plt.figure(figsize=(10, 8))
        fig.suptitle('Stand_results')
        ax = fig.add_subplot(111)
        plt.boxplot(results_)
        ax.set_xticklabels(pipelines.keys())
        plt.show()

    def specificmodel(self):
        # 参数优化 KNN

        scaler = StandardScaler().fit(X_train)
        rescaledX = scaler.transform(X_train)
        param_grid = {'n_neighbors': [1, 3, 5, 7, 9, 11, 13, 15, 17, 19, 21]}
        model = KNeighborsRegressor()
        kfold = KFold(n_splits=10, shuffle=False)
        grid = GridSearchCV(estimator=model, param_grid=param_grid, scoring='neg_mean_squared_error', cv=kfold)
        grid_result = grid.fit(rescaledX, y_train)
        print('best param is : %s, and score is : %f' % (grid_result.best_params_, grid_result.best_score_))
        cv_results = zip(grid_result.cv_results_['mean_test_score'], grid_result.cv_results_['std_test_score'], grid_result.cv_results_['params'])
        for mean, std, param in cv_results:
            print(param, mean, std)