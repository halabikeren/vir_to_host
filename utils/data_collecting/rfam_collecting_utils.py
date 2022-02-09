import os
import shutil
import tarfile
import typing as t
import mysql
import numpy as np
import pandas as pd
from serivces.mysql_service import MySQLService


class RfamCollectingUtils:
    db_connection: mysql.connector.connection_cext.CMySQLConnection

    def __init__(self):
        # connect to public rfam db using details in https://docs.rfam.org/en/latest/database.html
        self.db_connection = mysql.connector.connect(
            user="rfamro", host="mysql-rfam-public.ebi.ac.uk", port="4497", database="Rfam"
        )

    def __del__(self):
        self.db_connection.close()

    def get_virus_name_id_mapping(self, output_path: str,) -> pd.DataFrame:
        """
        :param output_path: path to write the output dataframe to
        :return: the output dataframe
        """

        if os.path.exists(output_path):
            return pd.read_csv(output_path)

        virus_id_name_query = 'SELECT ncbi_id, species FROM taxonomy WHERE tax_string LIKE "%Viruses%";'
        rfam_virus_name_to_virus_id_df = pd.read_sql(sql=virus_id_name_query, con=self.db_connection)
        rfam_virus_name_to_virus_id_df.to_csv(output_path, index=False)

        return rfam_virus_name_to_virus_id_df

    def get_id_to_rfamseq_acc_mapping(
        self, query_rfam_viruses_ids: t.List[str], query_batch_size: int, output_path: str,
    ) -> pd.DataFrame:
        """
        :param query_rfam_viruses_ids: viruses ids to query on
        :param query_batch_size: batch size for sqlite queries
        :param output_path: path to write the output dataframe to
        :return: the output dataframe
        :return:
        """

        if os.path.exists(output_path):
            return pd.read_csv(output_path)

        query_template = "SELECT DISTINCT tx.ncbi_id, rf.rfamseq_acc FROM taxonomy tx, rfamseq rf WHERE tx.ncbi_id IN ({}) AND rf.ncbi_id = tx.ncbi_id;"
        query_workdir = f"{os.path.dirname(output_path)}/rfam_acc_queries/"
        rfam_virus_ids_to_rfamseq_acc_df = MySQLService.do_batch_query(
            connection=self.db_connection,
            query_template=query_template,
            query_items=query_rfam_viruses_ids,
            workdir=query_workdir,
            batch_size=query_batch_size,
        )

        rfam_virus_ids_to_rfamseq_acc_df.to_csv(output_path, index=False)
        shutil.rmtree(query_workdir, ignore_errors=True)

        return rfam_virus_ids_to_rfamseq_acc_df

    def get_rfamseq_acc_to_rfam_acc_mapping(
        self, query_rfamseq_acc_ids: t.List[str], query_batch_size: int, output_path: str,
    ) -> pd.DataFrame:
        """
        :param query_rfamseq_acc_ids: rfamseq accessions to query on
        :param query_batch_size: batch size for sqlite queries
        :param output_path: path to write the output dataframe to
        :return: the output dataframe
        :return:
        """

        if os.path.exists(output_path):
            return pd.read_csv(output_path)

        query_template = "SELECT DISTINCT rf.rfamseq_acc, fr.rfam_acc FROM rfamseq rf, full_region fr WHERE rf.rfamseq_acc IN ({}) AND rf.rfamseq_acc = fr.rfamseq_acc AND fr.is_significant = 1;"
        query_workdir = f"{os.path.dirname(output_path)}/rfam_seq_acc_queries/"
        rfamseq_acc_to_rfam_acc_df = MySQLService.do_batch_query(
            connection=self.db_connection,
            query_template=query_template,
            query_items=query_rfamseq_acc_ids,
            workdir=query_workdir,
            batch_size=query_batch_size,
        )

        rfamseq_acc_to_rfam_acc_df.to_csv(output_path, index=False)
        shutil.rmtree(query_workdir, ignore_errors=True)

        return rfamseq_acc_to_rfam_acc_df

    def get_rfam_acc_to_rfam_id_mapping(
        self, query_rfam_acc_ids: t.List[str], query_batch_size: int, output_path: str,
    ) -> pd.DataFrame:
        """
        :param query_rfam_acc_ids: rfam accessions to query on
        :param query_batch_size: batch size for sqlite queries
        :param output_path: path to write the output dataframe to
        :return: the output dataframe
        :return:
        """

        if os.path.exists(output_path):
            return pd.read_csv(output_path)

        query_template = "SELECT DISTINCT fr.rfam_acc, fm.rfam_id, fm.description FROM full_region fr, family fm WHERE fr.rfam_acc IN ({}) AND fr.rfam_acc = fm.rfam_acc;"
        query_workdir = f"{os.path.dirname(output_path)}/rfam_id_queries/"
        rfam_acc_to_rfam_id_df = MySQLService.do_batch_query(
            connection=self.db_connection,
            query_template=query_template,
            query_items=query_rfam_acc_ids,
            workdir=query_workdir,
            batch_size=query_batch_size,
        )

        rfam_acc_to_rfam_id_df.to_csv(output_path, index=False)
        shutil.rmtree(query_workdir, ignore_errors=True)

        return rfam_acc_to_rfam_id_df

    def get_viral_rfam_data(self, output_path: str) -> pd.DataFrame:
        """
        :param output_path: path ot write the collected data to
        :return:
        """

        output_dir = os.path.dirname(output_path)
        os.makedirs(output_dir, exist_ok=True)

        rfam_virus_name_to_virus_id_df = self.get_virus_name_id_mapping(
            output_path=f"{output_dir}/rfam_virus_name_to_virus_id.csv",
        )

        rfam_virus_ids = [str(int(item)) for item in rfam_virus_name_to_virus_id_df["ncbi_id"].unique()]
        rfam_virus_ids_to_rfamseq_acc_df = self.get_id_to_rfamseq_acc_mapping(
            query_rfam_viruses_ids=rfam_virus_ids,
            output_path=f"{output_dir}/rfam_virus_ids_to_rfamseq_acc.csv",
            query_batch_size=1000,
        )

        df = rfam_virus_name_to_virus_id_df.merge(rfam_virus_ids_to_rfamseq_acc_df, on="ncbi_id", how="left")

        rfamseq_acc_ids = ['"' + item + '"' for item in rfam_virus_ids_to_rfamseq_acc_df["rfamseq_acc"].unique()]
        rfamseq_acc_to_rfam_acc_df = self.get_rfamseq_acc_to_rfam_acc_mapping(
            query_rfamseq_acc_ids=rfamseq_acc_ids,
            output_path=f"{output_dir}/rfamseq_acc_to_rfam_acc.csv",
            query_batch_size=1000,
        )

        df = df.merge(rfamseq_acc_to_rfam_acc_df, on="rfamseq_acc", how="left")

        rfam_acc_ids = ['"' + item + '"' for item in rfamseq_acc_to_rfam_acc_df["rfam_acc"].unique()]
        rfam_acc_to_rfam_id_df = self.get_rfam_acc_to_rfam_id_mapping(
            query_rfam_acc_ids=rfam_acc_ids, output_path=f"{output_dir}/rfam_acc_to_rfam_id.csv", query_batch_size=1000,
        )

        df = df.merge(rfam_acc_to_rfam_id_df, on="rfam_acc", how="left")

        relevant_df = df[["ncbi_id", "species", "rfam_acc"]].rename(
            columns={"ncbi_id": "species_id", "species": "species_name"}
        )

        relevant_df.dropna(subset=["rfam_acc"], how="any", axis=0, inplace=True)
        relevant_df.drop_duplicates(inplace=True)
        relevant_df.to_csv(f"{output_dir}/rfam_data.csv", index=False)

        return relevant_df

    @staticmethod
    def get_cm_models(required_rfam_ids: t.List[str], output_dir: str, wget_path: str):
        """
        :param required_rfam_ids: rfam ids to get alignments for
        :param output_dir: directory to write the alignments to
        :param wget_path: wget url of the cm models
        :return: none
        """
        os.makedirs(output_dir, exist_ok=True)
        output_paths = [f"{output_dir}/{rfam_id}.cm" for rfam_id in required_rfam_ids]
        cm_models_available = np.all([os.path.exists(output_path) for output_path in output_paths])
        if not cm_models_available:
            zipped_output_path = f"{os.getcwd()}/{os.path.basename(wget_path)}"
            if not os.path.exists(zipped_output_path):
                res = os.system(f"wget {wget_path}")
            # unzip output to output_dir
            with tarfile.open(zipped_output_path, "r:gz") as file:
                file.extractall(output_dir)
            for path in os.listdir(output_dir):
                if path.replace(".cm", "") not in required_rfam_ids:
                    os.remove(f"{output_dir}/{path}")
