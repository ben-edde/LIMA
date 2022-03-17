from sklearn.model_selection import cross_validate, TimeSeriesSplit

class Dataset:
    def __init__(self,X=None,y=None,train_ratio=1): 
        self.feature=X
        self.label=y
        self.train_ratio=train_ratio
        self.update()
    
    def set_train_ratio(self,train_ratio):
        self.train_ratio=train_ratio
        self.update()
    
    def update(self):
        self.length=self.feature.shape[0]
        train_size=int(self.length*self.train_ratio)
        self.train_X=self.feature[:train_size]
        self.train_y=self.label[:train_size]
        self.test_X=self.feature[train_size:]
        self.test_y=self.label[train_size:]
 
    def get_TS_cv(self, k=5, test_size=None): 
        return TimeSeriesSplit(
            n_splits=k,
            gap=0,
            test_size=test_size,
        )