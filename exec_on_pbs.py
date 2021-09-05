import json
import os
import typing as t
import sys
from asyncio import sleep

import click

import logging

import pandas as pd
import numpy as np

logger = logging.getLogger(__name__)


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
#PBS -q {priority}
#PBS -v PBS_O_SHELL=bash,PBS_ENVIRONMENT=PBS_BATCH
#PBS -N {job_name}
#PBS -e {job_output_dir}
#PBS -o {job_output_dir}
#PBS -l select=ncpus={cpus_num}:mem={ram_gb_size}gb
{commands_str}
"""
    with open(job_path, "w") as outfile:
        outfile.write(job_content)

    return 0


@click.command()
@click.option(
    "--df_input_path",
    type=click.Path(exists=True, file_okay=True, readable=True),
    help="path holding the original dataframe to fragment",
)
@click.option(
    "--df_output_path",
    type=click.Path(exists=False, file_okay=True, readable=True),
    help="path holding the output dataframe to write",
)
@click.option(
    "--batch_size",
    type=click.INT,
    help="the requirement number of records in each dataframe fragment to work on",
    default=5000,
)
@click.option(
    "--execution_type",
    type=click.IntRange(0, 1),
    help="sequential (0) or parallelized (1)",
    default=0,
)
@click.option(
    "--workdir",
    type=click.Path(exists=False, dir_okay=True, writable=True),
    help="directory to operate in",
)
@click.option(
    "--job_cpus_num",
    type=click.INT,
    help="number of cpus to use in each job",
    default=1,
)
@click.option(
    "--job_ram_gb_size",
    type=click.INT,
    help="size of memory in gb to use in each job",
    default=4,
)
@click.option(
    "--job_priority",
    type=click.INT,
    help="number of cpus to use in each job",
    default=0,
)
@click.option(
    "--job_queue", type=click.STRING, help="queue to submit jobs to", default="itaym"
)
@click.option(
    "--script_to_exec",
    type=click.Path(exists=True, file_okay=True, readable=True),
    help="absolute path to the script to execute",
)
@click.option(
    "--script_input_path_argname",
    type=click.STRING,
    help="name of argument of script input path",
    default="input_path",
)
@click.option(
    "--script_output_path_argname",
    type=click.STRING,
    help="name of argument of script output path",
    default="output_path",
)
@click.option(
    "--script_log_path_argname",
    type=click.STRING,
    help="name of argument of script logger path",
    default="logger_path",
)
@click.option(
    "--script_default_args_json",
    type=click.Choice([click.Path(exists=True, file_okay=True, readable=True), None]),
    help="path ot json with default script args",
    required=False,
    default=None,
)
def exe_on_pbs(
    df_input_path: click.Path,
    df_output_path: click.Path,
    batch_size: int,
    execution_type: int,
    workdir: click.Path,
    job_cpus_num: int,
    job_ram_gb_size: int,
    job_priority: int,
    job_queue: str,
    script_to_exec: click.Path,
    script_input_path_argname: str,
    script_output_path_argname: str,
    script_log_path_argname: str,
    script_default_args_json: t.Optional[click.Path],
):

    # initialize the logger
    logger_path = f"{workdir}/{__name__}.log"
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s module: %(module)s function: %(funcName)s line: %(lineno)d %(message)s",
        handlers=[
            logging.StreamHandler(sys.stdout),
            logging.FileHandler(logger_path),
        ],
    )

    # set working environment
    os.makedirs(workdir, exist_ok=True)
    input_dfs_dir = f"{workdir}/input_dfs/"
    os.makedirs(input_dfs_dir, exist_ok=True)
    output_dfs_dir = f"{workdir}/output_dfs/"
    os.makedirs(output_dfs_dir, exist_ok=True)
    logs_dir = f"{workdir}/logs/"
    os.makedirs(logs_dir, exist_ok=True)
    jobs_dir = f"{workdir}/jobs/"
    os.makedirs(jobs_dir, exist_ok=True)
    jobs_output_dir = f"{workdir}/jobs_output/"
    os.makedirs(jobs_output_dir, exist_ok=True)
    logger.info(f"working environment for execution pipeline created in {workdir}")

    # create input dfs
    input_df = pd.read_csv(df_input_path)
    dfs_num = int(input_df.shape[0] / batch_size)
    input_sub_dfs = np.array_split(input_df, dfs_num)
    input_sub_dfs_paths = []
    for i in range(len(input_sub_dfs)):
        sub_df_path = f"{input_dfs_dir}df_{i}.csv"
        input_sub_dfs[i].to_csv(sub_df_path, index=False)
        input_sub_dfs_paths.append(sub_df_path)
    logger.info(
        f"written {dfs_num} input dataframes of size {batch_size} to {input_dfs_dir}"
    )

    # create job files
    script_dir = os.path.dirname(script_to_exec)
    script_filename = os.path.basename(script_to_exec)
    default_args = ""
    if script_default_args_json and os.path.exists(script_default_args_json):
        with open(script_default_args_json, "rb") as infile:
            default_args_dict = json.load(infile)
        default_args += " ".join(
            [
                f"--{argname}={default_args_dict[argname]}"
                for argname in default_args_dict
            ]
        )
    job_path_to_output_path = dict()
    for i in range(len(input_sub_dfs)):
        job_name = f"{script_filename}_{i}"
        job_path = f"{jobs_dir}{job_name}.sh"
        job_output_dir = f"{jobs_output_dir}{script_filename}_{i}/"
        os.makedirs(job_output_dir, exist_ok=True)
        input_path = input_sub_dfs_paths[i]
        output_path = f"{output_dfs_dir}{os.path.basename(input_path)}"
        logger_path = f"{logs_dir}{script_filename}_{i}.log"
        commands = [
            f"cd {script_dir}",
            f"python {script_filename} {default_args} --{script_input_path_argname}={input_path} --{script_output_path_argname}={output_path} --{script_log_path_argname}={logger_path}",
        ]
        res = create_job_file(
            job_path=job_path,
            job_name=job_name,
            job_output_dir=jobs_output_dir,
            commands=commands,
            queue=job_queue,
            priority=job_priority,
            cpus_num=job_cpus_num,
            ram_gb_size=job_ram_gb_size,
        )
        job_path_to_output_path[job_path] = output_path
    jobs_paths = list(job_path_to_output_path.keys())
    logger.info(f"creation of {len(jobs_paths)} in {jobs_dir} is complete")

    # submit jobs based on the chosen type of execution
    logger.info(
        f"submitting jobs in {'sequential' if execution_type == 0 else 'parallelized'} mode"
    )
    if execution_type == 0:  # sequential
        job_index = 0
        while job_index < len(jobs_paths):
            job_path = jobs_paths[job_index]
            res = os.system(f"qsub {job_path}")
            job_output_path = job_path_to_output_path[job_path]
            while not os.path.exists(job_output_path):
                sleep(20)
            logger.info(f"job {job_index} is complete")
            job_index += 1
    else:  # parallelized
        for job_path in jobs_paths:
            res = os.system(f"qsub {job_path}")
        complete = all(
            [
                os.path.exists(job_output_path)
                for job_output_path in job_path_to_output_path.values()
            ]
        )
        while not complete:
            sleep(20)
            jobs_paths_exist = [
                os.path.exists(job_output_path)
                for job_output_path in job_path_to_output_path.values()
            ]
            complete = all(jobs_paths_exist)
            logger.info(f"{jobs_paths_exist.count(False)} jobs are not finished yet")
    logger.info("jobs execution is complete")

    # concat all the output dataframes to create a single output dataframe
    logger.info(
        f"concatenating {len(list(job_path_to_output_path.values()))} output dataframes into a single one"
    )
    output_dfs = [
        pd.read_csv(job_output_path)
        for job_output_path in job_path_to_output_path.values()
    ]
    output_df = pd.concat(output_dfs)
    output_df.to_csv(df_output_path, index=False)


if __name__ == "__main__":
    exe_on_pbs()
