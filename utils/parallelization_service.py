import logging
import os

logger = logging.getLogger(__name__)

import pandas as pd
import numpy as np

# from pandas_msgpack import read_msgpack

from multiprocessing import Pool
from functools import partial


class ParallelizationService:
    @staticmethod
    def parallelize(
        df: pd.DataFrame,
        func: partial.func,
        num_of_processes: int = 1,
    ):
        df_split = np.array_split(df, num_of_processes)
        pool = Pool(num_of_processes)
        df_paths = [path for path in pool.map(partial(func), df_split)]
        df = pd.concat([pd.read_csv(path) for path in df_paths])
        for path in df_paths:
            if os.path.exists(path):
                os.remove(path)
        pool.close()
        pool.join()
        return df

    @staticmethod
    def run_on_subset(func: partial.func, df_subset: pd.DataFrame):
        df = df_subset.apply(func, axis=1)
        return df

    @staticmethod
    def parallelize_on_rows(
        df: pd.DataFrame,
        func: partial.func,
        num_of_processes: int = 1,
    ):
        return ParallelizationService.parallelize(
            df,
            partial(ParallelizationService.run_on_subset, func),
            num_of_processes,
        )
