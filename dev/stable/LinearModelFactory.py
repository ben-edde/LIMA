from sklearn.linear_model import LinearRegression, Lasso, Ridge


class LinearModelFactory:
    @classmethod
    def get_model(self, model_name: str = "LR"):
        if model_name == "LR":
            return LinearRegression()
        elif model_name == "LASSO":
            return Lasso(random_state=42)
        elif model_name == "Ridge":
            return Ridge(random_state=42)