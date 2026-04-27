from typing import Protocol


class Predictor(Protocol):
    def predict_batch(self, items: list) -> list: ...


class AllFalsePredictor:
    """Always returns False. Used for the day-one deployable stub."""
    def predict_batch(self, items: list) -> list:
        return [False] * len(items)
