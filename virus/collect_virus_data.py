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

from utils.data_collecting_utils import (
    SequenceCollectingUtils,
    GenomeBiasCollectingService,
)


class ViralFeature(Enum):
    SecStruct = 1
    PRF = 2
    ELM = 3
    Pheno = 4


@click.command()
@click.option(
    "--virus_data_path",
    type=click.Path(exists=True, file_okay=True, readable=True),
    help="path of the virus data, along with previously collected sequence data",
    default="./data/virus_data.csv",
)
@click.option(
    "--associations_data_path",
    type=click.Path(exists=True, file_okay=True, readable=True),
    help="path of the association data, holding the virus taxonomic information",
    default="./data/associations_united.csv",
)
@click.option(
    "--previous_studies_dir",
    type=click.Path(exists=True, file_okay=True, readable=True),
    help="directory that holds the data collected from previous studies",
    default="./data/previous_studies/virus/",
)
@click.option(
    "--database_sources_dir",
    type=click.Path(exists=True, dir_okay=True),
    help="directory holding associations data from databases",
    default="./data/databases/virus/",
)
@click.option(
    "--inferred_sources_dir",
    type=click.Path(exists=True, dir_okay=True),
    help="directory to hold predicted or inferred viral features from sequence data",
    default="./data/inferred/virus/",
)
@click.option(
    "--logger_path",
    type=click.Path(exists=False, file_okay=True),
    help="path to logging file",
    default="./data/collect_virus_data.log",
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
    virus_data_path: click.Path,
    associations_data_path: click.Path,
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
    virus_sequence_data = pd.read_csv(virus_data_path)

    # extract taxonomic features from the associations file
    associations_data = pd.read_csv(associations_data_path)
    virus_taxonomic_data = associations_data[
        [col for col in associations_data.columns if "virus_" in col]
    ]
    virus_taxonomic_data.drop_duplicates(
        subset=["virus_taxon_name", "virus_taxon_id"], inplace=True
    )
    virus_data = virus_sequence_data.merge(
        on=["virus_taxon_name", "virus_taxon_id"], how="left"
    )

    # compute genomic features using the available sequence data
    # virus_genomic_bias_data = virus_data[""] # by refseq and then by genbank and then by gi

    # collect secondary RNA structural data from existing databases and predict missing data using viennarna: https://www.tbi.univie.ac.at/RNA/ViennaRNA/doc/html/examples_python.html


if __name__ == "__main__":
    collect_viral_features()
