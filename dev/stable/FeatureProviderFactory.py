from NewsFeatureProvider import NewsFeatureProvider
from PriceFeatureProvider import PriceFeatureProvider


class FeatureProviderFactory:
    @classmethod
    def get_provider(self, category: str):
        if category == "news":
            return NewsFeatureProvider()
        elif category == "price":
            return PriceFeatureProvider()
