import os
import shutil
import gzip
import sys
import typing as t

import mysql.connector
from Bio import SeqIO
from Bio.Seq import Seq
from Bio.SeqRecord import SeqRecord

from utils.data_collecting_utils import MySQLUtils
from utils.pbs_utils import PBSUtils

import click
import logging

import pandas as pd

logger = logging.getLogger(__name__)


def get_virus_name_id_mapping(
    connection: mysql.connector.connection_cext.CMySQLConnection, output_path: str,
) -> pd.DataFrame:
    """
    :param connection: db connection to rfam db
    :param output_path: path to write the output dataframe to
    :return: the output dataframe
    """

    if os.path.exists(output_path):
        return pd.read_csv(output_path)

    virus_id_name_query = 'SELECT ncbi_id, species FROM taxonomy WHERE tax_string LIKE "%Viruses%";'
    rfam_virus_name_to_virus_id_df = pd.read_sql(sql=virus_id_name_query, con=connection)
    rfam_virus_name_to_virus_id_df.to_csv(output_path, index=False)

    return rfam_virus_name_to_virus_id_df


def get_id_to_rfamseq_acc_mapping(
    connection: mysql.connector.connection_cext.CMySQLConnection,
    query_rfam_viruses_ids: t.List[str],
    query_batch_size: int,
    output_path: str,
) -> pd.DataFrame:
    """
    :param connection: db connection to rfam db
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
    rfam_virus_ids_to_rfamseq_acc_df = MySQLUtils.do_batch_query(
        connection=connection,
        query_template=query_template,
        query_items=query_rfam_viruses_ids,
        workdir=query_workdir,
        batch_size=query_batch_size,
    )

    rfam_virus_ids_to_rfamseq_acc_df.to_csv(output_path, index=False)
    shutil.rmtree(query_workdir, ignore_errors=True)

    return rfam_virus_ids_to_rfamseq_acc_df


def get_rfamseq_acc_to_rfam_acc_mapping(
    connection: mysql.connector.connection_cext.CMySQLConnection,
    query_rfamseq_acc_ids: t.List[str],
    query_batch_size: int,
    output_path: str,
) -> pd.DataFrame:
    """
    :param connection: db connection to rfam db
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
    rfamseq_acc_to_rfam_acc_df = MySQLUtils.do_batch_query(
        connection=connection,
        query_template=query_template,
        query_items=query_rfamseq_acc_ids,
        workdir=query_workdir,
        batch_size=query_batch_size,
    )

    rfamseq_acc_to_rfam_acc_df.to_csv(output_path, index=False)
    shutil.rmtree(query_workdir, ignore_errors=True)

    return rfamseq_acc_to_rfam_acc_df


def get_rfam_acc_to_rfam_id_mapping(
    connection: mysql.connector.connection_cext.CMySQLConnection,
    query_rfam_acc_ids: t.List[str],
    query_batch_size: int,
    output_path: str,
) -> pd.DataFrame:
    """
    :param connection: db connection to rfam db
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
    rfam_acc_to_rfam_id_df = MySQLUtils.do_batch_query(
        connection=connection,
        query_template=query_template,
        query_items=query_rfam_acc_ids,
        workdir=query_workdir,
        batch_size=query_batch_size,
    )

    rfam_acc_to_rfam_id_df.to_csv(output_path, index=False)
    shutil.rmtree(query_workdir, ignore_errors=True)

    return rfam_acc_to_rfam_id_df


def get_viral_rfam_data(output_path: str) -> pd.DataFrame:
    """
    :param output_path: path ot write the collected data to
    :return:
    """
    # connect to public rfam db using details in https://docs.rfam.org/en/latest/database.html
    connection = mysql.connector.connect(
        user="rfamro", host="mysql-rfam-public.ebi.ac.uk", port="4497", database="Rfam"
    )

    output_dir = os.path.dirname(output_path)
    os.makedirs(output_dir, exist_ok=True)

    rfam_virus_name_to_virus_id_df = get_virus_name_id_mapping(
        connection=connection, output_path=f"{output_dir}/rfam_virus_name_to_virus_id.csv",
    )

    rfam_virus_ids = [str(int(item)) for item in rfam_virus_name_to_virus_id_df["ncbi_id"].unique()]
    rfam_virus_ids_to_rfamseq_acc_df = get_id_to_rfamseq_acc_mapping(
        connection=connection,
        query_rfam_viruses_ids=rfam_virus_ids,
        output_path=f"{output_dir}/rfam_virus_ids_to_rfamseq_acc.csv",
        query_batch_size=1000,
    )

    df = rfam_virus_name_to_virus_id_df.merge(rfam_virus_ids_to_rfamseq_acc_df, on="ncbi_id", how="left")

    rfamseq_acc_ids = ['"' + item + '"' for item in rfam_virus_ids_to_rfamseq_acc_df["rfamseq_acc"].unique()]
    rfamseq_acc_to_rfam_acc_df = get_rfamseq_acc_to_rfam_acc_mapping(
        connection=connection,
        query_rfamseq_acc_ids=rfamseq_acc_ids,
        output_path=f"{output_dir}/rfamseq_acc_to_rfam_acc.csv",
        query_batch_size=1000,
    )

    df = df.merge(rfamseq_acc_to_rfam_acc_df, on="rfamseq_acc", how="left")

    rfam_acc_ids = ['"' + item + '"' for item in rfamseq_acc_to_rfam_acc_df["rfam_acc"].unique()]
    rfam_acc_to_rfam_id_df = get_rfam_acc_to_rfam_id_mapping(
        connection=connection,
        query_rfam_acc_ids=rfam_acc_ids,
        output_path=f"{output_dir}/rfam_acc_to_rfam_id.csv",
        query_batch_size=1000,
    )

    df = df.merge(rfam_acc_to_rfam_id_df, on="rfam_acc", how="left")

    relevant_df = df[["ncbi_id", "species", "rfam_acc"]].rename(
        columns={"ncbi_id": "species_id", "species": "species_name"}
    )

    relevant_df.dropna(subset=["rfam_acc"], how="any", axis=0, inplace=True)
    relevant_df.drop_duplicates(inplace=True)
    relevant_df.to_csv(f"{output_dir}/rfam_data.csv", index=False)

    connection.close()

    return relevant_df


def write_sequence_db(sequence_data_path: str, seq_db_path: str, db_species_names: t.List[str]):
    """
    :param sequence_data_path: dataframe with sequence data
    :param seq_db_path:path to write the database to
    :param db_species_names:species whose accessions should be included in the database
    :return: none
    """
    if not os.path.exists(sequence_data_path):
        sequence_df = pd.read_csv(sequence_data_path)
        relevant_sequence_df = sequence_df.loc[sequence_df.species_name.isin(db_species_names)]
        db_records = []
        for i, row in relevant_sequence_df.iterrows():
            db_records.append(
                SeqRecord(id=row.accession, name=row.accession, description=row.accession, seq=Seq(row.sequence))
            )
        SeqIO.write(db_records, seq_db_path, format="fasta")


def write_rfam_alignments(required_rfam_ids: t.List[str], output_dir: str, wget_dir: str):
    """
    :param required_rfam_ids: rfam ids to get alignments for
    :param output_dir: directory to write the alignments to
    :param wget_dir: wget url of the alignments
    :return: none
    """
    os.makedirs(output_dir, exist_ok=True)
    if len(os.listdir(output_dir)) < len(required_rfam_ids):
        curr_dir = os.getcwd()
        os.chdir(output_dir)
        for rfam_id in required_rfam_ids:
            wget_path = f"{wget_dir}{rfam_id}.fa.gz"
            if not os.path.exists(f"{output_dir}{rfam_id}.fa.gz"):
                res = os.system(f"wget {wget_path}")
        os.chdir(curr_dir)
        # unzip all paths
        for path in os.listdir(output_dir):
            if ".fa.gz" in path:
                with gzip.open(f"{output_dir}{path}", "rb") as f_in:
                    with open(f"{output_dir}{path.replace('.fa.gz', '.fasta')}", "wb") as f_out:
                        shutil.copyfileobj(f_in, f_out)
                os.remove(f"{output_dir}{path}")


def apply_rfam_based_search(alignments_dir: str, workdir: str, output_dir: str, db_path: str):
    """
    :param alignments_dir: directory of files of aligned sequences in fasta format
    :param workdir: path for write the jobs of the pipeline per alignment in
    :param output_dir: directory to write the pipeline outputs on the alignment files in
    :param db_path: path to sequence db file
    :return: none
    """
    parent_path = (
        f"'{os.path.dirname(os.getcwd())}'"
        if "pycharm" not in os.getcwd()
        else "'/groups/itay_mayrose/halabikeren/vir_to_host/'"
    )
    os.makedirs(workdir, exist_ok=True)
    os.makedirs(output_dir, exist_ok=True)
    for path in os.listdir(alignments_dir):
        rfam_id = path.replace(".fasta", "")
        aln_workdir = f"{workdir}/{rfam_id}/"
        os.makedirs(aln_workdir, exist_ok=True)
        job_name = f"cmsearch_{rfam_id}"
        job_path = f"{aln_workdir}/{job_name}.sh"
        cmd_alignment_path = f"'{alignments_dir}{path}'"
        cmd_workdir = f"'{aln_workdir}'"
        cmd_output_dir = f"'{output_dir}/{rfam_id}/'"
        cmd_db_path = f"'{db_path}'"
        cmd = f'python -c "import sys;sys.path.append({parent_path});from utils.rna_struct_utils import RNAStructUtils;RNAStructUtils.apply_infernal_search_on_alignment(alignment_path={cmd_alignment_path}, workdir={cmd_workdir}, output_dir={cmd_output_dir}, db_path={cmd_db_path})"'
        PBSUtils.create_job_file(
            job_name=job_name, job_output_dir=aln_workdir, job_path=job_path, commands=[cmd], cpus_num=2, ram_gb_size=10
        )
        os.system(f"qsub {job_path}")


@click.command()
@click.option(
    "--rfam_data_path",
    type=click.Path(exists=False, file_okay=True, readable=True),
    help="path to dataframe with rfam ids corresponding to viral secondary structures and the viral species name they are associated with",
    required=False,
    default=None,
)
@click.option(
    "--associations_data_path",
    type=click.Path(exists=True, file_okay=True, readable=True),
    help="path to virus-host associations data, based on which the 'phenotype' of each category (columns) will be determined",
    required=True,
)
@click.option(
    "--sequence_data_path",
    type=click.Path(exists=True, file_okay=True, readable=True),
    help="path to viral sequence dataframe",
    required=True,
)
@click.option(
    "--virus_taxonomic_filter_column_name",
    type=click.Choice(["genus", "family", "class"]),
    help="taxonomic rank to select viral data for analysis based on",
    required=False,
    default="family",
)
@click.option(
    "--virus_taxonomic_filter_column_value",
    type=str,
    help="taxonomic rank to select viral data for analysis based on",
    required=False,
    default="flaviviridae",
)
@click.option(
    "--host_taxonomic_group",
    type=click.Choice(["species", "genus", "family", "class"]),
    help="path to virus-host associations data, based on which the 'phenotype' of each category (columns) will be determined",
    required=False,
    default="class",
)
@click.option(
    "--workdir",
    type=click.Path(exists=False, file_okay=True, readable=True),
    help="directory to hold the pipeline files in",
    required=False,
    default=None,
)
@click.option(
    "--rfam_wget_dir",
    type=str,
    help="url to the rfam fasta ftp services",
    required=False,
    default="https://ftp.ebi.ac.uk/pub/databases/Rfam/CURRENT/fasta_files/",
)
@click.option(
    "--log_path",
    type=click.Path(exists=False, file_okay=True, readable=True),
    help="path holding the logging of the script",
)
@click.option(
    "--output_dir",
    type=click.Path(exists=False, file_okay=True, readable=True),
    help="path holding the output of the association tests",
)
@click.option(
    "--multiple_test_correction_method",
    type=click.Choice(["bh", "bonferroni"]),
    help="method for correction for multiple testing across categories (rows)",
    required=False,
    default="bh",
)
def test_structs_host_associations(
    rfam_data_path: str,
    associations_data_path: str,
    sequence_data_path: str,
    virus_taxonomic_filter_column_name: t.Optional[str],
    virus_taxonomic_filter_column_value: str,
    host_taxonomic_group: str,
    workdir: str,
    rfam_wget_dir: str,
    log_path: str,
    output_dir: str,
    multiple_test_correction_method: str,
):
    # initialize the logger
    os.makedirs(workdir, exist_ok=True)
    os.makedirs(output_dir, exist_ok=True)
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s module: %(module)s function: %(funcName)s line %(lineno)d: %(message)s",
        handlers=[logging.StreamHandler(sys.stdout), logging.FileHandler(str(log_path)),],
        force=True,  # run over root logger settings to enable simultaneous writing to both stdout and file handler
    )

    # process input data
    associations_df = pd.read_csv(associations_data_path)
    relevant_associations_df = associations_df
    if virus_taxonomic_filter_column_name is not None:
        relevant_associations_df = associations_df.loc[
            associations_df[f"virus_{virus_taxonomic_filter_column_name}_name"] == virus_taxonomic_filter_column_value
        ]
    viral_species_names = list(relevant_associations_df.virus_species_name.unique())
    logger.info(
        f"processed {relevant_associations_df.shape[0]} associations across {len(viral_species_names)} viral species for analysis"
    )

    # collect rfam data, if not already available
    if rfam_data_path is None or not os.path.exists(rfam_data_path):
        logger.info(f"extracting viral RFAM data ('genotype') data")
        if rfam_data_path is None:

            rfam_data_path = f"{output_dir}/rfam_data.csv"
        rfam_data = get_viral_rfam_data(output_path=rfam_data_path)
    else:
        rfam_data = pd.read_csv(rfam_data_path)
    logger.info(f"collected data of {len(list(rfam_data.rfam_acc.unique()))} rfam records")

    # create a sequence "database" in the form a a single giant fasta file (for downstream cmsearch executions)
    seq_db_path = f"{workdir}/sequence_database.fasta"
    write_sequence_db(
        sequence_data_path=sequence_data_path, seq_db_path=seq_db_path, db_species_names=viral_species_names
    )
    logger.info(f"wrote sequence database to {sequence_data_path}")

    # for each unique rfam id, get the alignment that conferred it from the rfam ftp service: http://ftp.ebi.ac.uk/pub/databases/Rfam/CURRENT/fasta_files/
    rfam_workdir = f"{workdir}/rfam/"
    rfam_alignments_dir = f"{rfam_workdir}alignments/"
    write_rfam_alignments(
        required_rfam_ids=list(rfam_data.rfam_acc.unique()), output_dir=rfam_alignments_dir, wget_dir=rfam_wget_dir,
    )
    logger.info(f"downloaded rfam alignments to {rfam_alignments_dir}")

    # apply rfam based search on each alignment
    apply_rfam_based_search(
        alignments_dir=rfam_alignments_dir,
        workdir=f"{rfam_workdir}search_pipeline/",
        output_dir=f"{workdir}/cmsearch_results/",
        db_path=seq_db_path,
    )
    logger.info(
        f"submitted infernal pipeline jobs for the {len(rfam_alignments_dir)} collected alignments against the sequence db at {seq_db_path}"
    )


if __name__ == "__main__":
    test_structs_host_associations()
