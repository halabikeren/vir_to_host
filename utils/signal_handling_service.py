import logging

logger = logging.getLogger(__name__)

import pandas as pd


class SignalHandlingService:
    @staticmethod
    def exit_handler(df: pd.DataFrame, output_path: str, signum):
        """
        :param df: dataframe currently worked on
        :param output_path: path to write the dataframe to
        :param signum: signal number
        :return: none
        """
        logger.error(f"closing on signal {signum} and saving temporary output to {output_path}")
        df.to_csv(path_or_buf=output_path, index=False)
        exit(0)
