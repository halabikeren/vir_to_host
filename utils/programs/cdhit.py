import os
import re
import typing as t

from Bio import SeqIO
from Bio.Seq import Seq
from Bio.SeqRecord import SeqRecord

from settings import get_settings
import numpy as np
import pandas as pd
import logging

logger = logging.getLogger(__name__)


class CdHit:
    @staticmethod
    def exec_cdhit(input_path: str, output_dir: str, homology_threshold: float = 0.95) -> str:
        """
  :param input_path: path to unaligned sequences in fasta format
  :param output_dir: directory to create output in
  :param homology_threshold: threshold for cdhit
  :return:
  """

        logger.info(f"cdhit input paths created at {output_dir}")
        cdhit_output_prefix = f"{output_dir}/cdhit_out_thr_{homology_threshold}"
        cdhit_log_file = f"{output_dir}/cdhit.log"
        if not os.path.exists(f"{cdhit_output_prefix}.clstr"):
            word_len = (
                (8 if homology_threshold > 0.7 else 4)
                if homology_threshold > 0.6
                else (3 if homology_threshold > 0.5 else 2)
            )
            logger.info(
                f"executing cdhit on {input_path} with homology threshold of {homology_threshold} and word length {word_len}"
            )
            cmd = f"{get_settings().CDHIT_DIR}cd-hit-est -i {input_path} -o {cdhit_output_prefix} -c {homology_threshold} -n {word_len} -M {output_dir} > {cdhit_log_file}"
            res = os.system(cmd)
            if res != 0:
                raise RuntimeError(f"CD-HIT failed to properly execute and provide an output file with error")

        return cdhit_output_prefix

    @staticmethod
    def get_largest_cdhit_cluster(sequence_records: t.List[SeqRecord], workdir: str, homology_threshold: float = 0.99):
        """
  :param sequence_records: aligned sequence records
  :param workdir: directory to execute cdhit on unaligned records in, and select the ones in the largest cluster
  :param homology_threshold: threshold for cdhit execution
  :return: the accessions of the records within the largest cluster
  """
        os.makedirs(workdir, exist_ok=True)
        cdhit_input_path = f"{workdir}/cdhit_input.fasta"
        unaligned_sequence_records = sequence_records
        for record in unaligned_sequence_records:
            record.seq = Seq(str(record.seq).replace("-", ""))
        SeqIO.write(unaligned_sequence_records, cdhit_input_path, format="fasta")
        cdhit_output_prefix = CdHit.exec_cdhit(
            input_path=cdhit_input_path, output_dir=workdir, homology_threshold=homology_threshold
        )
        clusters = CdHit.get_cdhit_cluster_members(clusters_path=f"{cdhit_output_prefix}.clstr")
        return max(clusters, key=len)

    @staticmethod
    def collapse_redundant_sequences(
        elements: pd.DataFrame,
        homology_threshold: t.Optional[float] = 0.99,
        aux_dir: str = f"{os.getcwd()}/cdhit_aux/",
        mem_limit: int = 4000,
    ):
        """
 :param elements: elements to cluster using cdhit for the purpose of removing redundancy using cdhit
 :param homology_threshold: cdhit threshold in removing redundant sequences
 :param aux_dir: directory to write cdhit output files to
 :param mem_limit: memory allocation for cdhit
 :return: none, adds a column of "sequence_representative" to each column, with the accession selected by cdhit as the cluster representative
 as the sequences within each cluster are at least 99% similar, the choice of the cluster representative doesn't have to be wise
 """
        logger.info(
            f"removing redundancy across {elements.shape[0]} elements using cd-hit with a threshold of {homology_threshold}"
        )

        elm_to_cluster = CdHit.get_cdhit_clusters(
            elements=elements,
            homology_threshold=homology_threshold,
            aux_dir=aux_dir,
            memory_limit=mem_limit,
            return_cdhit_cluster_representative=True,
        )

        accession_regex = re.compile("(.*?)_\D")
        elements["sequence_representative"] = np.nan
        accession_to_cluster = {accession_regex.search(elm).group(1): elm_to_cluster[elm] for elm in elm_to_cluster}
        elements.set_index("accession", inplace=True)
        elements["sequence_representative"].fillna(value=accession_to_cluster, inplace=True)
        elements.reset_index(inplace=True)
        logger.info(f"representative of redundant sequences have been recorded")

    @staticmethod
    def get_cdhit_clusters(elements, homology_threshold, aux_dir, memory_limit, return_cdhit_cluster_representative):
        pass

    @staticmethod
    def get_cdhit_cluster_members(clusters_path: str) -> t.List[t.List[str]]:
        """
        :param clusters_path: oath of cdhit clustering output
        :return: a list of cluster members within each cluster id (which corresponds to the list index)
        """
        with open(clusters_path, "r") as infile:
            clusters_data = [item.split("\n") for item in infile.read().split(">Cluster ")[1:]]
        cluster_member_regex = re.compile(">(\w*).")
        clusters = []
        for data in clusters_data:
            cluster_members = [cluster_member_regex.search(item).group(1) for item in data[1:-1]]
            clusters.append(cluster_members)
        return clusters
