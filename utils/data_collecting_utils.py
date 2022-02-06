import logging
import os

import typing as t

logger = logging.getLogger(__name__)
import mysql.connector

import pandas as pd
from Bio import Entrez

sys.path.append("..")
from settings import get_settings

Entrez.email = get_settings().ENTREZ_EMAIL


class DataCleanupUtils:
    @staticmethod
    def handle_duplicated_columns(colname: str, df: pd.DataFrame) -> pd.DataFrame:
        """
        :param colname: name of the column that was duplicated as a result of the merge
        :param df: dataframe to remove duplicated columns from
        :return:
        """
        duplicated_columns = [col for col in df.columns if colname in col and col != colname]
        if len(duplicated_columns) == 0:
            return df
        main_colname = duplicated_columns[0]
        for col in duplicated_columns[1:]:

            # check that there are no contradictions
            contradictions = df.dropna(subset=[main_colname, col], how="any", inplace=False)
            contradictions = contradictions.loc[contradictions[main_colname] != contradictions[col]]
            if contradictions.shape[0] > 0:
                logger.error(
                    f"{contradictions.shape[0]} contradictions found between column values in {main_colname} and {col}. original column values will be overridden"
                )
                df.loc[df.index.isin(contradictions.index), main_colname] = df.loc[
                    df.index.isin(contradictions.index), main_colname
                ].apply(lambda x: contradictions.loc[contradictions[main_colname] == x, col].values[0])
            df[main_colname] = df[main_colname].fillna(df[col])
        df = df.rename(columns={main_colname: colname})
        for col in duplicated_columns[1:]:
            df = df.drop(col, axis="columns")
        return df


class MySQLUtils:
    @staticmethod
    def do_batch_query(
        connection: mysql.connector.connection_cext.CMySQLConnection,
        query_template: str,
        query_items: t.List[str],
        workdir: str,
        batch_size: int = 1000,
    ) -> pd.DataFrame:
        """

        :param connection: connection to mysql db
        :param query_template: string of type 'XXX {}', on which .format(item) operation can be applied
        :param workdir: directory to write intermediate batch query results to
        :param query_items: items to divide into batches for the query
        :param batch_size: batch size for queries
        :return: dataframe of complete query
        """

        os.makedirs(workdir, exist_ok=True)
        final_queries_result_path = f"{workdir}/complete_query.csv"
        if os.path.exists(final_queries_result_path):
            return pd.read_csv(final_queries_result_path)

        batches = [query_items[i : i + batch_size] for i in range(0, len(query_items), batch_size)]
        queries = [query_template.format(",".join(batch)) for batch in batches]
        queries_results = []
        for i in range(len(queries)):
            try:
                query_result_path = f"{workdir}/query_{i}.csv"
                if not os.path.exists(query_result_path):
                    query_result = pd.read_sql_query(sql=queries[i], con=connection)
                    query_result.to_csv(query_result_path, index=False)
                else:
                    query_result = pd.read_csv(query_result_path)
                queries_results.append(query_result)
            except Exception as e:
                logger.error(f"failed to retrieve data for query {i} due to error {e}, and will thus skip it")
                continue
        queries_result = pd.concat(queries_results)
        queries_result.to_csv(final_queries_result_path, index=False)
        for i in range(len(queries)):
            query_result_path = f"{workdir}/query_{i}.csv"
            if os.path.exists(query_result_path):
                os.remove(query_result_path)

        return queries_result
