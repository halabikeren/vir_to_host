import os
import re
import shutil
from time import sleep
import typing as t

import glob
import numpy as np
import pandas as pd
from Bio import SeqIO
from Bio.Seq import Seq
from Bio.SeqRecord import SeqRecord

from serivces import PBSService

import logging

logger = logging.getLogger(__name__)


class Infernal:
    sequence_db_path: str

    def __init__(self, sequence_db_path: str):
        self.sequence_db_path = sequence_db_path

    def write_sequence_db(self, sequence_data_dir: str):
        """
        :param sequence_data_dir: path to sequence data files ofr which structural regions were predicted
        :return: none
        """
        if not os.path.exists(self.sequence_db_path):
            sequence_data_paths = [
                path
                for path in glob.glob(sequence_data_dir + "/**/*.fasta", recursive=True)
                if "rnaz_candidates_mlocarna_aligned" in path
            ]
            db_records = []
            for path in sequence_data_paths:
                records = list(SeqIO.parse(path, format="fasta"))
                for record in records:
                    db_records.append(
                        SeqRecord(
                            id=f"{record.id}",
                            name=record.name,
                            description=record.description,
                            seq=Seq(str(record.seq).replace("-", "")),
                        )
                    )
            SeqIO.write(db_records, self.sequence_db_path, format="fasta")

    @staticmethod
    def apply_search(cm_models_dir: str, workdir: str, output_dir: str):
        """
        :param cm_models_dir: directory of covariance models of relevant rfam ids
        :param workdir: path for write the jobs of the pipeline per alignment in
        :param output_dir: directory to write the pipeline outputs on the cm models in
        :return: none
        """
        os.makedirs(workdir, exist_ok=True)
        os.makedirs(output_dir, exist_ok=True)
        for path in os.listdir(cm_models_dir):
            rfam_id = path.replace(".cm", "")
            search_output_dir = f"{output_dir}/{rfam_id}/"
            os.makedirs(search_output_dir, exist_ok=True)
            if len(os.listdir(search_output_dir)) == 0:
                rfam_workdir = f"{workdir}/{rfam_id}/"
                os.makedirs(rfam_workdir, exist_ok=True)
                cov_model_path = f"{cm_models_dir}/{path}"
                job_name = f"cmsearch_{rfam_id}"
                job_path = f"{rfam_workdir}/{job_name}.sh"
                cmd = f"cmsearch -A {search_output_dir}aligned_hits.fasta --tblout {search_output_dir}hits.tsv {cov_model_path} {self.sequence_db_path} > {search_output_dir}cmsearch.out"
                PBSService.create_job_file(
                    job_name=job_name,
                    job_output_dir=rfam_workdir,
                    job_path=job_path,
                    commands=[cmd],
                    cpus_num=2,
                    ram_gb_size=10,
                )
                os.system(f"qsub {job_path}")
        complete = np.all(
            [len(os.listdir(f"{workdir}/{path.replace('.cm', '')}/")) > 1 for path in os.listdir(cm_models_dir)]
        )
        while not complete:
            sleep(2 * 60)
            complete = np.all(
                [len(os.listdir(f"{workdir}/{path.replace('.cm', '')}/")) > 1 for path in os.listdir(cm_models_dir)]
            )

    def apply_struct_search_pipeline(self, alignment_path: str, work_dir: str, output_dir: str) -> int:
        """
        :param alignment_path: path to alignment of suspected structural region
        :param work_dir: directory to write pipeline output to
        :param output_dir: directory to write final output to
        :return:
        """
        os.makedirs(work_dir, exist_ok=True)
        cov_model_path = f"{work_dir}/{os.path.basename(alignment_path).split('.')[0]}.cm"
        self.exec_cmbuild(alignment_path=alignment_path, cov_model_path=cov_model_path)
        self.exec_cmcalibrate(cov_model_path=cov_model_path)
        self.exec_cmsearch(cov_model_path=cov_model_path, search_output_dir=output_dir)

    @staticmethod
    def exec_cmbuild(alignment_path: str, cov_model_path: str):
        if not alignment_path.endswith("fasta"):
            alignment_format = alignment_path.split(".")[-1]
            records = list(SeqIO.parse(alignment_path, alignment_format))
            new_alignment_path = alignment_path.replace(alignment_format, "fasta")
            SeqIO.write(records, new_alignment_path, format="fasta")
            os.remove(alignment_path)
            alignment_path = new_alignment_path
        res = os.system(f"cmbuild --noss {cov_model_path} {alignment_path}")
        if res != 0:
            if os.path.exists(cov_model_path):
                os.remove(cov_model_path)
            error_msg = f"failed to execute cmbuild on {alignment_path}. output not written to {cov_model_path}"
            logger.error(error_msg)
            raise ValueError(error_msg)
        return res

    @staticmethod
    def exec_cmcalibrate(cov_model_path: str) -> int:
        backup_path = shutil.copyfile(cov_model_path, cov_model_path.replace(".cm", "pre_calibration.cm"))
        res = os.system(f"cmbalibrate {cov_model_path}")
        if res != 0:
            error_msg = f"failed to calibrate model at {cov_model_path}. will override with backup copy"
            os.remove(cov_model_path)
            os.rename(backup_path, cov_model_path)
            logger.error(error_msg)
            raise ValueError(error_msg)
        return res

    def exec_cmsearch(self, cov_model_path: str, search_output_dir: str) -> int:
        os.makedirs(search_output_dir, exist_ok=True)
        res = os.system(
            f"cmsearch -A {search_output_dir}aligned_hits.fasta --tblout {search_output_dir}hits.tsv {cov_model_path} {self.sequence_db_path} > {search_output_dir}cmsearch.out"
        )
        if res != 0:
            error_msg = f"failed to execute cmsearch on {cov_model_path}. output not written or written partially to {search_output_dir}"
            logger.error(error_msg)
            raise ValueError(error_msg)
        return res

    @staticmethod
    def get_hits(
        ids: t.List[str], search_results_dir: str, hit_id_to_required_id_map: t.Dict[str, str]
    ) -> t.Dict[str, t.List[str]]:
        """
        :param ids: ids of rfam for which infernal search has been applied
        :param search_results_dir: oath to the results of infernal search
        :param hit_id_to_required_id_map: map of accessions to their respective species
        :return: map fo rfam ids to their hit species
        """
        query_id_to_mapped_hits = dict()
        acc_regex = re.compile("(.*?)\/", re.DOTALL)
        for query_id in ids:
            hits_table_path = f"{search_results_dir}/{query_id}/hits.tsv"
            hits_alignment_path = f"{search_results_dir}/{query_id}/aligned_hits.fasta"
            if (
                os.path.exists(hits_table_path)
                and os.path.exists(hits_alignment_path)
                and os.stat(hits_alignment_path).st_size > 0
            ):
                hits = pd.read_csv(hits_table_path, sep="\s+", skiprows=[1])
                hits_ids = [acc_regex.search(hit_id).group(1) for hit_id in hits.loc[hits["#target"] != "#", "#target"]]
                mapped_hits_ids = list(set([hit_id_to_required_id_map[acc] for acc in hits_ids]))
                query_id_to_mapped_hits[query_id] = mapped_hits_ids
        return query_id_to_mapped_hits

    @staticmethod
    def infer_covariance_models(alignments_dir: str, covariance_models_dir: str, workdir: str):
        """
        :param alignments_dir: directory of alignments based on which covariance models will be inferred
        :param covariance_models_dir: directory to write the inferred covariance models to
        :param workdir: directory of pbs jobs ans their outputs
        :return:
        """
        cmds = [
            "source /groups/itay_mayrose/halabikeren/.bashrc",
            "cmbuild --noss {output_path} {input_path}",
        ]
        PBSService.execute_job_array(
            input_dir=alignments_dir,
            output_dir=covariance_models_dir,
            work_dir=workdir,
            output_format="cm",
            commands=cmds,
            commands_argnames_to_varnames={"input_path": "input_path", "output_path": "output_path"},
            input_paths_suffix=".fasta",
        )

    @staticmethod
    def calibrate_covariance_models(covariance_models_dir: str, workdir: str):
        cmds = [
            "source /groups/itay_mayrose/halabikeren/.bashrc",
            "cmcalibrate {input_path}",
        ]
        PBSService.execute_job_array(
            input_dir=covariance_models_dir,
            output_dir=covariance_models_dir,
            work_dir=workdir,
            output_format="cm",
            commands=cmds,
            commands_argnames_to_varnames={"input_path": "input_path", "output_path": "output_path"},
        )
