import os

from pydantic import BaseSettings
from functools import lru_cache
from pathlib import Path


class Settings(BaseSettings):
    ENTREZ_API_KEY: str

    class Config:
        env_file = Path(__file__).parent.joinpath(".env").absolute()


@lru_cache()
def get_settings() -> Settings:
    return Settings()
