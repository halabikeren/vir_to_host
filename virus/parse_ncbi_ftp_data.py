import multiprocessing
import os
import sys
import typing as t
import numpy as np
import pandas as pd
from Bio import SeqIO

sys.path.append("..")
from serivces.parallelization_service import ParallelizationService

ftp_data_dir = "/groups/itay_mayrose/halabikeren/vir_to_host/data/databases/ncbi_viral_seq_data/"
output_path = "/groups/itay_mayrose/halabikeren/vir_to_host/data/databases/ncbi_viral_seq_data.csv"


def parse_ncbi_gb_data(gb_paths: pd.DataFrame) -> str:
    """
    :param gb_paths: df with paths to genbank records
    :return: dataframe with genbank records data
    """
    df_path = f"{os.getcwd()}/df_{parse_ncbi_gb_data.__name__}_pid_{os.getpid()}.csv"

    ncbi_df = pd.DataFrame(
        columns=[
            "taxon_id",
            "accession",
            "source",
            "sequence",
            "annotation",
            "cds",
            "keywords",
            "category",
            "accession_genome_index",
        ]
    )

    for path in pd.Series(gb_paths).values:
        record_data = dict()
        record = list(SeqIO.parse(path, format="genbank"))[0]
        record_data["accession"] = record.id.split(".")[0]
        record_data["sequence"] = str(record.seq)
        record_data["annotation"] = record.description
        record_data["keywords"] = ";".join(record.__dict__["annotations"]["keywords"])
        cds_regions = []
        for feature in record.features:
            if feature.type == "source":
                record_data["taxon_id"] = "".join(feature.qualifiers["db_xref"])
            if feature.type == "CDS":
                start = int(feature.__dict__["location"]._start)
                end = int(feature.__dict__["location"]._end)
                strand = int(feature.__dict__["location"]._strand)
                location = f"{'complement(' if strand == -1 else ''}{start}..{end}{')' if strand == -1 else ''}"
                cds_regions.append(location)
        record_data["cds"] = ";".join(cds_regions)
        accession_genome_index = np.nan
        annotation = record_data["annotation"]
        if "DNA-N" in annotation or "DNA-C" in annotation:
            accession_genome_index = 0
        elif "DNA-P" in annotation or "DNA-E2" in annotation:
            accession_genome_index = 1
        elif "DNA-M" in annotation or "DNA-E3" in annotation:
            accession_genome_index = 2
        elif "DNA-G" in annotation:
            accession_genome_index = 3
        elif "DNA-L" in annotation:
            accession_genome_index = 3
        record_data["accession_genome_index"] = accession_genome_index

        ncbi_df = ncbi_df.append(record_data, ignore_index=True)

    ncbi_df.to_csv(df_path, index=False)
    return df_path


if __name__ == "__main__":

    paths = [f"{ftp_data_dir}{path}" for path in os.listdir(ftp_data_dir)]
    paths_df = pd.DataFrame(pd.Series(paths))

    df = ParallelizationService.parallelize(
        df=paths_df, func=parse_ncbi_gb_data(), num_of_processes=multiprocessing.cpu_count() - 1,
    )

    df.to_csv()
