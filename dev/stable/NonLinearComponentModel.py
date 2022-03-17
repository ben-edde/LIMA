from statsmodels.tsa.seasonal import STL

class NonLinearComponentModel:
    def __init__(self,dataset) -> None:
        self.dataset=dataset
        self.model=None
    
    def stationary_check(self):
        pass
    def differecing(self):
        if not self.stationary_check():            
            self.dataset.feature=np.diff(self.dataset.feature,axis=0)
            # original_non_linear_y=non_linear_y[:-1]
            self.dataset.label=np.diff(self.dataset.label,axis=0)
        self.label=np.concatenate([self.non_linear_trend,self.non_linear_season,self.non_linear_residual],axis=1)
        
            
    def stl_decompose(self):
        stl = STL(self.dataset.label,10)
        res = stl.fit()
        self.non_linear_trend=res.trend.reshape(-1,1)
        self.non_linear_season=res.seasonal.reshape(-1,1)
        self.non_linear_residual=res.resid.reshape(-1,1)
        # decomposed_y=np.array([res.trend,res.seasonal,res.resid]).transpose()
        # non_linear_y=decomposed_y
        # fig = res.plot()

    def set_model(self,model_name="LR"):
        self.model=LinearModelFactory.get_model(model_name)

    def train(self):
        train_X=self.dataset.train_X
        train_y=self.dataset.train_y.ravel()
        self.model.fit(train_X,train_y)

    def predict(self,feature): 
        return self.model.predict(feature)