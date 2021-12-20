import json
import os
import re
import typing as t
import sys
from time import sleep

import click

import logging

import pandas as pd
import numpy as np

from utils.pbs_utils import PBSUtils

logger = logging.getLogger(__name__)





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
    "--split_input_by",
    type=click.Choice(["column", "size"]),
    help="indicator weather input should be split by size or by column values for a given column",
    default="size",
)
@click.option(
    "--batch_size",
    type=click.INT,
    help="the requirement number of records in each dataframe fragment to work on. used only if split_input_by==size",
    required=False,
    default=5000,
)
@click.option(
    "--split_column",
    type=click.STRING,
    help="the column based on which the input df should be segmented (segment pair unique value in column). used only if split_input_by==column",
    required=False,
    default="",
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
    "--job_queue", type=click.STRING, help="queue to submit jobs to", default="itaymr"
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
    type=click.Path(exists=False, file_okay=True, readable=True),
    help="path to json with default script args",
    required=False,
    default=None,
)
@click.option(
    "--max_jobs_in_parallel",
    type=click.IntRange(1, 2000),
    help="path ot json with default script args",
    required=False,
    default=1900,
)
@click.option(
    "--output_suffix",
    type=str,
    help="path ot json with default script args",
    required=False,
    default="csv",
)
def exe_on_pbs(
    df_input_path: click.Path,
    df_output_path: click.Path,
    split_input_by: str,
    batch_size: int,
    split_column: str,
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
    max_jobs_in_parallel: int,
    output_suffix: str,
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

    # create input dfs, if they don't already exist
    if not os.listdir(input_dfs_dir):
        input_df = pd.read_csv(df_input_path)
        if split_input_by == "size":
            dfs_num = int(input_df.shape[0] / batch_size)
            input_sub_dfs = np.array_split(input_df, dfs_num)
            logger.info(
                f"writing {dfs_num} sub-dataframes of size {batch_size} to {input_dfs_dir}"
            )
        else:
            grouped_df = input_df.groupby(split_column)
            group_names = list(input_df[split_column].unique())
            input_sub_dfs = [
                grouped_df.get_group(group_name) for group_name in group_names
            ]
            dfs_num = len(input_sub_dfs)
            logger.info(
                f"writing {dfs_num} sub-dataframes of varying sizes to {input_dfs_dir}"
            )
        input_sub_dfs_paths = []
        for i in range(len(input_sub_dfs)):
            name = i
            if split_input_by == "column":
                name = re.sub('[^0-9a-zA-Z]+', '_', list(grouped_df.groups.keys())[i])
            sub_df_path = f"{input_dfs_dir}{name}.csv"
            input_sub_dfs[i].to_csv(sub_df_path, index=False)
            input_sub_dfs_paths.append(sub_df_path)
        logger.info(
            f"written {dfs_num} input dataframes of size {batch_size} to {input_dfs_dir}"
        )
    else:
        input_sub_dfs_paths = [
            f"{input_dfs_dir}{path}"
            for path in os.listdir(input_dfs_dir)
            if ".csv" in path
        ]
        dfs_num = len(input_sub_dfs_paths)
        logger.info(
            f"{dfs_num} sub-dataframes of of size {batch_size} of the original input dataframe are in {input_dfs_dir}"
        )

    # create job files
    script_dir = os.path.dirname(str(script_to_exec))
    script_filename = os.path.basename(str(script_to_exec))
    default_args = ""
    if script_default_args_json and os.path.exists(str(script_default_args_json)):
        with open(str(script_default_args_json), "rb") as infile:
            default_args_dict = json.load(infile)
        default_args += " ".join(
            [
                f"--{argname}={default_args_dict[argname]}"
                for argname in default_args_dict
            ]
        )
    job_path_to_output_path = dict()
    for i in range(dfs_num):
        name = i
        if split_input_by == "column":
            name = list(grouped_df.groups.keys())[i]
        job_name = f"{script_filename.split('.')[0]}_{name}"
        input_path = input_sub_dfs_paths[i]
        output_path = f"{output_dfs_dir}{os.path.basename(input_path).replace('.csv', f'.{output_suffix}')}"
        job_path = f"{jobs_dir}{job_name}.sh"
        if not os.path.exists(output_path):
            job_output_dir = f"{jobs_output_dir}{job_name}/"
            os.makedirs(job_output_dir, exist_ok=True)
            logger_path = f"{logs_dir}{job_name}.log"
            commands = [
                f"cd {script_dir}",
                f"python {script_filename} {default_args} --{script_input_path_argname}={input_path} --{script_output_path_argname}={output_path} --{script_log_path_argname}={logger_path}",
            ]
            res = PBSUtils.create_job_file(
                job_path=job_path,
                job_name=job_name,
                job_output_dir=job_output_dir,
                commands=commands,
                queue=job_queue,
                priority=job_priority,
                cpus_num=job_cpus_num,
                ram_gb_size=job_ram_gb_size,
            )
        job_path_to_output_path[job_path] = output_path
    jobs_paths = [job_path for job_path in job_path_to_output_path if not os.path.exists(job_path_to_output_path[job_path])]
    logger.info(f"creation of {len(jobs_paths)} in {jobs_dir} is complete")

    # submit jobs based on the chosen type of execution
    logger.info(
        f"submitting jobs in {'sequential' if execution_type == 0 else 'parallelized'} mode"
    )
    if execution_type == 0:  # sequential
        job_index = 0
        while job_index < len(jobs_paths):
            job_path = jobs_paths[job_index]
            job_output_path = job_path_to_output_path[job_path]
            if not os.path.exists(job_output_path):
                res = os.system(f"qsub {job_path}")
                logger.info(f"job {job_index} has been submitted")
                while not os.path.exists(job_output_path):
                    sleep(20)
                logger.info(f"job {job_index} is complete")
            job_index += 1
    else:  # parallelized
        for job_path in jobs_paths:
            # check how many jobs are running
            curr_jobs_num = PBSUtils.compute_curr_jobs_num()
            while curr_jobs_num > max_jobs_in_parallel:
                sleep(120)
                curr_jobs_num = PBSUtils.compute_curr_jobs_num()
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
    if output_suffix == "csv":
        output_dfs = [
            pd.read_csv(job_output_path)
            for job_output_path in job_path_to_output_path.values()
        ]
        output_df = pd.concat(output_dfs)
        output_df.to_csv(df_output_path, index=False)


if __name__ == "__main__":
    exe_on_pbs()
