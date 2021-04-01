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
from sklearn.neighbors import KNeighborsRegressor
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.ensemble import AdaBoostRegressor
from sklearn.metrics import mean_squared_error
from linearmodel import Linear


LIn = Linear()
# 数据理解
X_train, X_test, y_train, y_test = LIn.datatreating()

class Boostingmodelsss:


    def __init__(self):
        pass

    def modeleset(self):


        # 接着使用集成算法的模型

        ensembles = {}
        ensembles['ScalerAB'] = Pipeline([('Scaler', StandardScaler()), ('AB', AdaBoostRegressor())])
        ensembles['ScalerAB-KNN'] = Pipeline([('Scaler', StandardScaler()), ('ABKNN', AdaBoostRegressor(base_estimator=KNeighborsRegressor(n_neighbors=3)))])
        ensembles['ScalerAB-LR'] = Pipeline([('Scaler', StandardScaler()), ('ABLR', AdaBoostRegressor(LinearRegression()))])
        ensembles['ScalerRFR'] = Pipeline([('Scaler', StandardScaler()), ('RFR', RandomForestRegressor())])
        ensembles['ScalerETR'] = Pipeline([('Scaler', StandardScaler()), ('ETR', ExtraTreesRegressor())])
        ensembles['ScalerRBR'] = Pipeline([('Scaler', StandardScaler()), ('RBR', GradientBoostingRegressor())])

        results__ = []

        for key in ensembles:
            kfold = KFold(n_splits=10, shuffle=False)
            cv__result = cross_val_score(ensembles[key], X_train, y_train, scoring='neg_mean_squared_error', cv=kfold)
            results__.append(cv__result)

            print('%s: %f (%f)' % (key, cv__result.mean(), cv__result.std()))

    def fitting1(self):
        # 集成算法调参随机梯度和极端梯度
        scaler = StandardScaler().fit(X_train)
        rescalerdX = scaler.transform(X_train)
        param_grid = {'n_estimators': [10, 50, 100, 200, 300, 400, 500, 600, 700, 800, 900]}
        model = GradientBoostingRegressor()
        kfold = KFold(n_splits=10, shuffle=False)
        grid = GridSearchCV(model, param_grid=param_grid, scoring='neg_mean_squared_error', cv=kfold)
        grid__result = grid.fit(rescalerdX, y_train)

        print('%s : %f ' % (grid__result.best_params_, grid__result.best_score_))

    def fitting2(self):
        scaler = StandardScaler().fit(X_train)
        rescalerdX = scaler.transform(X_train)
        param_grid = {'n_estimators': [5, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100]}
        model = ExtraTreesRegressor()
        kfold = KFold(n_splits=10, shuffle=False)
        grid = GridSearchCV(model, param_grid=param_grid, scoring='neg_mean_squared_error', cv=kfold)
        grid__result = grid.fit(rescalerdX, y_train)

        print('%s : %f ' % (grid__result.best_params_, grid__result.best_score_))


        # fig = plt.figure(figsize=(10, 8))
        # fig.suptitle('bestselect_boosting')
        # ax = fig.add_subplot(121)
        # plt.boxplot(grid__result)
        # ax.set_xticklabels(ensembles.keys())
        # plt.show()
    def final_model(self):
        scaler = StandardScaler().fit(X_train)
        rescaledX = scaler.transform(X_train)
        gbr = ExtraTreesRegressor(n_estimators=50)
        gbr.fit(rescaledX, y_train)

        rescaledX_test = scaler.transform(X_test)
        predictions = gbr.predict(rescaledX_test)
        print(mean_squared_error(y_test, predictions))