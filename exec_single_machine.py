import json
import os
import typing as t
import sys

import click

import logging

import pandas as pd
import numpy as np

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
    "--batch_size",
    type=click.INT,
    help="the requirement number of records in each dataframe fragment to work on",
    default=5000,
)
@click.option(
    "--workdir",
    type=click.Path(exists=False, dir_okay=True, writable=True),
    help="directory to operate in",
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
    help="path ot json with default script args",
    required=False,
    default=None,
)
def exe_on_single_machine(
    df_input_path: click.Path,
    df_output_path: click.Path,
    batch_size: int,
    workdir: click.Path,
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

    # create input dfs, if they don't already exist
    if not os.listdir(input_dfs_dir):
        input_df = pd.read_csv(df_input_path)
        dfs_num = int(input_df.shape[0] / batch_size)
        input_sub_dfs = np.array_split(input_df, dfs_num)
        input_sub_dfs_paths = []
        logger.info(
            f"writing {dfs_num} sub-dataframes of size {batch_size} to {input_dfs_dir}"
        )
        for i in range(len(input_sub_dfs)):
            sub_df_path = f"{input_dfs_dir}df_{i}.csv"
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
    job_name_to_output_path = dict()
    job_name_to_command = dict()
    for i in range(dfs_num):
        job_name = f"{script_filename.split('.')[0]}_{i}"
        input_path = input_sub_dfs_paths[i]
        output_path = f"{output_dfs_dir}{os.path.basename(input_path)}"
        logger_path = f"{logs_dir}{job_name}.log"
        command = f"python {script_filename} {default_args} --{script_input_path_argname}={input_path} --{script_output_path_argname}={output_path} --{script_log_path_argname}={logger_path}"
        job_name_to_output_path[job_name] = output_path
        job_name_to_command[job_name] = command
    jobs_names = list(job_name_to_command.keys())
    logger.info(f"computation of {len(jobs_names)} commands is complete")

    # submit jobs based on the chosen type of execution
    os.chdir(script_dir)
    logger.info(f"submitting jobs in sequential mode")
    for job_name in jobs_names:
        if not os.path.exists(job_name_to_output_path[job_name]):
            res = os.system(job_name_to_command[job_name])
            logger.info(f"job {job_name} is complete\n\n")

    logger.info("jobs execution is complete")

    # concat all the output dataframes to create a single output dataframe
    logger.info(
        f"concatenating {len(list(job_name_to_output_path.values()))} output dataframes into a single one"
    )
    output_dfs = [
        pd.read_csv(job_output_path)
        for job_output_path in job_name_to_output_path.values()
    ]
    output_df = pd.concat(output_dfs)
    output_df.to_csv(df_output_path, index=False)


if __name__ == "__main__":
    exe_on_single_machine()
