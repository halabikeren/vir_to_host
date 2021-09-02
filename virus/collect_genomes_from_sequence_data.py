import multiprocessing

import pandas as pd
import numpy as np
import os
import sys
import logging

logger = logging.getLogger(__name__)

sys.path.append("..")
from utils.parallelization_service import ParallelizationService
from utils.sequence_utils import SequenceCollectingUtils

input_path = f"{os.getcwd()}/../data/virus_data_united.csv"
output_path = f"{os.getcwd()}/../data/virus_genome_data.csv"

if __name__ == "__main__":

    df = pd.read_csv(input_path)

    only_genomes_df = ParallelizationService.parallelize(
        df=df,
        func=partial(
            SequenceCollectingUtils.categorize_sequences,
            data_prefix="virus",
            id_field="virus_taxon_name",
        ),
        num_of_processes=multiprocessing.cpu_count() - 1,
    )

    only_genomes_df.to_csv(output_path, index=False)
