from abc import ABC, abstractmethod
from src.model.dataset import Dataset

class RecommendationModel(ABC):

    def __init__(self) -> None:
        super().__init__()

    @abstractmethod
    def train(self, data: Dataset) -> None:
        pass

    @abstractmethod
    def predict(self, user: int, movie: int) -> float:
        pass

    @abstractmethod
    def rank(self, user: int, n: int) -> list[int]:
        pass