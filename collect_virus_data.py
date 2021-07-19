import sys
import os

import click
import typing as t
from enum import Enum
import json

import numpy as np
import pandas as pd
from habanero import Crossref
import logging
from Bio import Entrez


class ViralFeature(Enum):
    SecStruct = 1
    PRF = 2
    ELM = 3
    Pheno = 4


@click.command()
@click.option(
    "--samples_path",
    type=click.Path(exists=True, file_okay=True, readable=True),
    help="path of the association sampled from which the virus taxon ids should be extracted",
    default="./data/associations_united.csv",
)
@click.option(
    "--previous_studies_dir",
    type=click.Path(exists=True, file_okay=True, readable=True),
    help="directory that holds the data collected from previous studies",
    default="./data/previous_studies/associations/",
)
@click.option(
    "--database_sources_dir",
    type=click.Path(exists=True, dir_okay=True),
    help="directory holding associations data from databases",
    default="./data/databases/associations/",
)
@click.option(
    "--inferred_sources_dir",
    type=click.Path(exists=True, dir_okay=True),
    help="directory to hold predicted or inferred viral features from sequence data",
    default="./data/databases/inferred/",
)
@click.option(
    "--logger_path",
    type=click.Path(exists=False, file_okay=True),
    help="path to logging file",
    default="./data/collect_associations_data.log",
)
@click.option(
    "--debug_mode",
    type=click.BOOL,
    help="boolean indicating weather script should be executed in debug mode",
    default=False,
)
@click.option(
    "--output_path",
    type=click.Path(exists=False, file_okay=True),
    help="path to output file consisting of the united associations data",
    default="./data/viral_features_united.csv",
)
def collect_viral_features(
    samples_path: click.Path,
    previous_studies_dir: click.Path,
    database_sources_dir: click.Path,
    inferred_sources_dir: click.Path,
    logger_path: click.Path,
    debug_mode: bool,
    output_path: click.Path,
):
    # initialize the logger
    logging.basicConfig(
        level=logging.DEBUG if debug_mode else logging.INFO,
        format="%(asctime)s module: %(module)s function: %(funcName)s line: %(lineno)d %(message)s",
        handlers=[logging.StreamHandler(sys.stdout), logging.FileHandler(logger_path),],
    )
    logger = logging.getLogger(__name__)

    # get virus taxon ids and accessions, if available, from the samples data
    associations_df = pd.read_csv(samples_path)
    viral_features_df = associations_df[
        [col for col in associations_df.columns if "virus_" in col]
    ]

    # collect secondary RNA structural data in the form of membership in sub-vogs


if __name__ == "__main__":
    collect_viral_features()
