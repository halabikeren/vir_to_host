import logging
import os
import typing as t

import subprocess

logger = logging.getLogger(__name__)


class PBSUtils:
    @staticmethod
    def create_job_file(
        job_path,
        job_name: str,
        job_output_dir: str,
        commands: t.List[str],
        queue: str = "itaym",
        priority: int = 0,
        cpus_num: int = 1,
        ram_gb_size: int = 4,
    ) -> int:
        """
        :param job_path: absolute path to job file
        :param job_name: name of job
        :param job_output_dir: absolute path to job output files
        :param commands: list of commands to run from the job
        :param queue: queue to submit job to
        :param priority:  job's priority
        :param cpus_num: number of cpus to use
        :param ram_gb_size: size fo ram in gb to use
        :return: 0
        """
        os.makedirs(os.path.dirname(job_path), exist_ok=True)
        os.makedirs(os.path.dirname(job_output_dir), exist_ok=True)
        commands_str = "\n".join(commands)
        job_content = f"""# !/bin/bash
    #PBS -S /bin/bash
    #PBS -j oe
    #PBS -r y
    #PBS -q {queue}
    #PBS -p {priority}
    #PBS -v PBS_O_SHELL=bash,PBS_ENVIRONMENT=PBS_BATCH
    #PBS -N {job_name}
    #PBS -e {job_output_dir}
    #PBS -o {job_output_dir}
    #PBS -r y
    #PBS -l select=ncpus={cpus_num}:mem={ram_gb_size}gb
    {commands_str}
    """
        with open(job_path, "w") as outfile:
            outfile.write(job_content)

        return 0

    @staticmethod
    def compute_curr_jobs_num(username: str = "halabikeren") -> int:
        """
        :return: returns the current number of jobs
        """
        proc = subprocess.run(f"qselect -u {username} | wc -l", shell=True, check=True, capture_output=True)
        curr_jobs_num = int(proc.stdout)
        return curr_jobs_num
