from sklearn.svm import SVR, LinearSVR
from sklearn.tree import DecisionTreeRegressor
from sklearn.kernel_ridge import KernelRidge
from sklearn.linear_model import LinearRegression, ARDRegression, SGDRegressor, ElasticNet, Lars, Lasso, GammaRegressor, TweedieRegressor, PoissonRegressor, Ridge, BayesianRidge
from sklearn.ensemble import AdaBoostRegressor,RandomForestRegressor


class LinearComponentModel:
    def __init__(self) -> None:
        self.train_X=None
        self.train_y=None
        self.model=None
    def main(self):
        X_scaler = MinMaxScaler()
        X_scaler.fit(train_X)
        train_X=X_scaler.transform(train_X) 
        lin_model=LinearRegression()
        lin_model.fit(train_X,train_y.ravel())
        # linear_y=lin_model.predict(train_X)
        self.model=lin_model