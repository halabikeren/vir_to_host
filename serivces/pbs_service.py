import logging
import os
import re
import typing as t
import getpass
import subprocess
import shutil

import glob
from time import sleep

logger = logging.getLogger(__name__)


class PBSService:
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
    def compute_curr_jobs_num() -> int:
        """
        :return: returns the current number of jobs under the shell username
        """
        username = getpass.getuser()
        proc = subprocess.run(f"qselect -u {username} | wc -l", shell=True, check=True, capture_output=True)
        curr_jobs_num = int(proc.stdout)
        return curr_jobs_num

    @staticmethod
    def execute_job_array(
        input_dir: str,
        work_dir: str,
        output_dir: str,
        output_format: str,
        commands: t.List[str],
        commands_argnames_to_varnames: t.Dict[str, str],
        input_condition: t.Optional[t.Callable] = None,
        max_parallel_jobs: int = 1900,
        **kwargs,
    ):
        """
        :param input_dir: directory holding the input files
        :param work_dir: directory to create the .sh job files
        and to write their output files in
        :param output_dir: directory that will hold the output files
        :param  output_format: format (or file suffix) of output files
        :param commands: list of commands execute. These
        commands are expected to have only two arguments for editing: input_path and output_path. all other arguments
        must be fixed
        :param commands_argnames_to_varnames: map of command argnames within the format bracket to their variable name
        used in the formatting process
        :param input_condition: function to filter input files based on, for downstream execution
        :param max_parallel_jobs: maximal number of jobs to have submitted by the respective user at
        the same time
        :return: none
        """

        # set env: tree should be identical between input dir, work dir and output dir
        os.makedirs(work_dir, exist_ok=True)
        os.makedirs(output_dir, exist_ok=True)

        def ig_f(dir, files):
            return [f for f in files if os.path.isfile(os.path.join(dir, f))]

        shutil.copytree(input_dir, work_dir, ignore=ig_f, dirs_exist_ok=True)
        shutil.copytree(input_dir, output_dir, ignore=ig_f, dirs_exist_ok=True)

        # traverse input dir and for each create job file and job output dir with command that pushes output to
        # output file
        input_paths = [
            path
            for path in glob.glob(input_dir + "**/*", recursive=True)
            if os.path.isfile(path)
            and (input_condition(path) if input_condition is input_condition is not None else True)
        ]
        logger.info(f"# input paths to execute commands on = {len(input_paths)}")
        job_paths, job_output_paths = [], []
        for input_path in input_paths:
            output_path = re.sub("\..*", f".{output_format}", input_path.replace(input_dir, output_dir))
            if output_path.endswith("/"):
                os.makedirs(output_path, exist_ok=True)
            job_path = re.sub("\..*", ".sh", input_path.replace(input_dir, work_dir))
            job_name = os.path.basename(job_path).replace(".sh", "")
            job_output_path = re.sub("\..*", ".out", input_path.replace(input_dir, work_dir))
            formatting_args = [
                f"{argname}='{eval(commands_argnames_to_varnames[argname])}'"
                for argname in commands_argnames_to_varnames
            ]
            job_commands = [command.format(*formatting_args) for command in commands]
            PBSService.create_job_file(
                job_path=job_path, job_name=job_name, job_output_dir=job_output_path, commands=job_commands, **kwargs
            )
            job_paths.append(job_path)
            job_output_paths.append(job_output_path)

        logger.info(f"# jobs to submit = {len(job_paths)}")

        # submit all the jobs, while maintaining the limit number on parallel jobs
        job_index = 0
        while job_index < len(job_paths):
            while PBSService.compute_curr_jobs_num() > max_parallel_jobs:
                sleep(2 * 60)
            res = os.system(f"qsub {job_paths[job_index]}")
            job_index += 1

            if job_index % 500 == 0:
                logger.info(f"submitted {job_index} jobs thus far")

        # remove work dir
        shutil.rmtree(work_dir, ignore_errors=True)
