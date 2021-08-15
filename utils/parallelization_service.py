import logging

logger = logging.getLogger(__name__)

import pandas as pd
import numpy as np

from multiprocessing import Pool
from functools import partial


class ParallelizationService:
    @staticmethod
    def parallelize(
        df: pd.DataFrame,
        func: partial.func,
        output_path: str,
        num_of_processes: int = 1,
    ):
        """
        :param df: dataframe worked on
        :param func: function for parallelization
        :param output_path: path to write the dataframe to
        :param num_of_processes:  number of workers
        :return: the complete dataframe
        """
        df_split = np.array_split(df, num_of_processes)
        pool = Pool(num_of_processes)
        df = pd.concat(pool.map(partial(func, output_path), df_split))
        df.to_csv(path_or_buf=output_path, index=False)
        pool.close()
        pool.join()
        return df

    @staticmethod
    def run_on_subset(func: partial.func, df_subset: pd.DataFrame, output_path: str):
        """
        :param func: function to run on dataframe subset
        :param df_subset: dataframe subset to run on
        :param output_path: path to write the dataframe subset to
        :return: the compelte dataframe subset
        """
        df = df_subset.apply(func, axis=1)
        df.to_csv(output_path, index=False)
        return df

    @staticmethod
    def parallelize_on_rows(
        df: pd.DataFrame,
        func: partial.func,
        output_path: str,
        num_of_processes: int = 1,
    ):
        """
        :param df: dataframe to work on
        :param func: function to apply on rows
        :param output_path: path to write the dataframe to
        :param num_of_processes: number of workers
        :return: none
        """
        return ParallelizationService.parallelize(
            df,
            partial(ParallelizationService.run_on_subset, func, output_path),
            output_path,
            num_of_processes,
        )
