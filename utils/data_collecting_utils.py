import logging
logger = logging.getLogger(__name__)

import pandas as pd
from Bio import Entrez
Entrez.email = "halabikeren@mail.tau.ac.il"


class DataCleanupUtils:
    @staticmethod
    def handle_duplicated_columns(colname: str, df: pd.DataFrame) -> pd.DataFrame:
        """
        :param colname: name of the column that was duplicated as a result of the merge
        :param df: dataframe to remove duplicated columns from
        :return:
        """
        duplicated_columns = [
            col for col in df.columns if colname in col and col != colname
        ]
        if len(duplicated_columns) == 0:
            return df
        main_colname = duplicated_columns[0]
        for col in duplicated_columns[1:]:

            # check that there are no contradictions
            contradictions = df.dropna(
                subset=[main_colname, col], how="any", inplace=False
            )
            contradictions = contradictions.loc[
                contradictions[main_colname] != contradictions[col]
                ]
            if contradictions.shape[0] > 0:
                logger.error(
                    f"{contradictions.shape[0]} contradictions found between column values in {main_colname} and {col}. original column values will be overridden"
                )
                df.loc[df.index.isin(contradictions.index), main_colname] = df.loc[
                    df.index.isin(contradictions.index), main_colname
                ].apply(
                    lambda x: contradictions.loc[
                        contradictions[main_colname] == x, col
                    ].values[0]
                )
            df[main_colname] = df[main_colname].fillna(df[col])
        df = df.rename(columns={main_colname: colname})
        for col in duplicated_columns[1:]:
            df = df.drop(col, axis="columns")
        return df

