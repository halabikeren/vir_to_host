import os
import typing as t

import numpy as np
import pandas as pd
from ete3 import Tree

import logging

from statsmodels.stats.multitest import multipletests

logger = logging.getLogger(__name__)


class Gemma:
    @staticmethod
    def compute_kinship_matrix(tree: Tree, samples_to_include: t.List[str]):
        """
        :param tree: tree with the species of interest
        :param samples_to_include: sample ids to include in the data
        :return: none
        """
        kinship_matrix = pd.DataFrame(columns=samples_to_include, index=samples_to_include)
        for i in range(len(samples_to_include)):
            for j in range(i, len(samples_to_include)):
                if samples_to_include[i] == samples_to_include[j]:
                    kinship_matrix.loc[kinship_matrix.index == samples_to_include[i], samples_to_include[j]] = 0
                else:
                    dist = tree.get_distance(samples_to_include[i], samples_to_include[j])
                    kinship_matrix.loc[kinship_matrix.index == samples_to_include[i], samples_to_include[j]] = dist
                    kinship_matrix.loc[kinship_matrix.index == samples_to_include[j], samples_to_include[i]] = dist
        # convert distance values to similarity values by reducing the original values from the maximal distance
        max_dist = np.max(np.max(kinship_matrix, axis=1))
        kinship_matrix = max_dist - kinship_matrix
        normalized_kinship_matrix = (kinship_matrix - kinship_matrix.mean().mean()) / (
            kinship_matrix.max().max() - kinship_matrix.min().min()
        )
        return normalized_kinship_matrix

    @staticmethod
    def process_samples_data(pa_matrix: pd.DataFrame, samples_to_include: t.List[str]):
        """
        :param pa_matrix: test subjects presence absence matrix across samples
        :param samples_to_include: sample ids to include in the data
        :return: none
        """
        pa_matrix = pa_matrix[pa_matrix.index.isin(samples_to_include)]
        # duplicate pa rows od samples that appear multiple times in samples_to_include
        idx = [y for x in samples_to_include for y in pa_matrix.index[[pa_matrix.index.values == x]]]
        dup_pa_matrix = pa_matrix.loc[idx].reset_index(drop=True)
        dup_pa_matrix.sort_index(inplace=True)
        processed_pa_matrix = dup_pa_matrix.T
        col_names = processed_pa_matrix.columns.to_list()
        processed_pa_matrix["demi_allele_1"] = "A"
        processed_pa_matrix["demi_allele_2"] = "T"
        processed_pa_matrix = processed_pa_matrix[["demi_allele_1", "demi_allele_2"] + col_names]
        logger.info(
            f"processed samples elements data for {dup_pa_matrix.shape[0]} samples and {dup_pa_matrix.shape[1]} elements"
        )
        return processed_pa_matrix

    @staticmethod
    def process_samples_trait(
        samples_trait_data: pd.DataFrame, sample_id_name: str, trait_name: str, samples_to_include: t.List[str]
    ) -> t.List[pd.DataFrame]:
        """
        :param samples_trait_data: trait data of the samples
        :param sample_id_name: name of samples id column
        :param trait_name: name of column corresponding ot the trait of interest
        :param samples_to_include: sample ids to include in the data
        :return: the processed samples data
        """
        relevant_samples_trait_data = (
            samples_trait_data.loc[samples_trait_data[sample_id_name].isin(samples_to_include)][
                [sample_id_name, trait_name]
            ]
            .drop_duplicates()
            .dropna()
        )
        relevant_samples_trait_data.sort_values([sample_id_name, trait_name], inplace=True)
        relevant_samples_trait_data[trait_name] = pd.Categorical(relevant_samples_trait_data[trait_name])
        relevant_samples_trait_data[trait_name] = relevant_samples_trait_data[trait_name].cat.codes

        processed_samples_to_trait_values = (
            relevant_samples_trait_data[[sample_id_name, trait_name]]
            .groupby(sample_id_name)
            .agg({trait_name: lambda x: list(x)})[trait_name]
            .to_dict()
        )
        keys, values = zip(*processed_samples_to_trait_values.items())
        ncombos = np.product([len(v) for v in values])
        logger.info(
            f"number of combinations of unique host class to viral species = {ncombos}, and will thus opt to duplicate samples with more than one class"
        )
        relevant_samples_trait_data.set_index(sample_id_name, inplace=True)
        return relevant_samples_trait_data

    @staticmethod
    def apply_lmm_association_test(
        pa_matrix: pd.DataFrame,
        samples_trait_data: pd.DataFrame,
        sample_id_name: str,
        trait_name: str,
        tree_path: str,
        output_dir: str,
        multiple_test_correction_method: str,
    ) -> pd.DataFrame:
        """
        :param pa_matrix: presence absence matrix of elements to test
        :param samples_trait_data: dataframe with map of samples to trait value
        :param sample_id_name:name of sample identifier column
        :param trait_name:name of trait identifier column
        :param tree_path: path to tree based on which a kinship matrix will be computed
        :param output_dir: directory in which gemma output will be written
        :param multiple_test_correction_method method for correcting for multiple tests
        :return: dataframe of the association test results
        """

        os.makedirs(output_dir, exist_ok=True)
        kinship_matrix_path = f"{output_dir}/kinship_matrix.csv"
        samples_data_path = f"{output_dir}/samples_pa_data.csv"
        samples_trait_path = f"{output_dir}/samples_trait_data.csv"
        test_results_suffix = "gemma_test_result"
        results_path = f"{output_dir}output/{test_results_suffix}.assoc.txt"

        if not os.path.exists(results_path):

            if (
                not os.path.exists(kinship_matrix_path)
                or not os.path.exists(samples_data_path)
                or not os.path.exists(samples_trait_path)
            ):
                tree = Tree(tree_path)
                kinship_samples = tree.get_leaf_names()
                pa_samples = pa_matrix.index.values
                trait_samples = samples_trait_data[sample_id_name].values
                intersection_species = [
                    sample for sample in kinship_samples if sample in pa_samples and sample in trait_samples
                ]
                logger.info(f"{len(intersection_species)} were found to have genotype, trait and kinship data")

                if not os.path.exists(samples_trait_path):
                    samples_trait_data = Gemma.process_samples_trait(
                        samples_trait_data=samples_trait_data,
                        sample_id_name=sample_id_name,
                        trait_name=trait_name,
                        samples_to_include=intersection_species,
                    )
                    samples_trait_data.to_csv(samples_trait_path.replace(".csv", "_unprocessed.csv"))
                    samples_trait_data.to_csv(samples_trait_path, index=False, header=False)
                    samples_trait_data.reset_index(inplace=True)
                else:
                    samples_trait_data = pd.read_csv(samples_trait_path.replace(".csv", "_unprocessed.csv"))

                samples_to_include = samples_trait_data[sample_id_name].values  # account for duplicated samples in
                # the following datasets to create

                if not os.path.exists(kinship_matrix_path):
                    kinship_matrix = Gemma.compute_kinship_matrix(tree=tree, samples_to_include=samples_to_include)
                    kinship_matrix.to_csv(kinship_matrix_path.replace(".csv", "_unprocessed.csv"))
                    kinship_matrix.to_csv(kinship_matrix_path, index=False, header=False)

                if not os.path.exists(samples_data_path):
                    samples_pa_matrix = Gemma.process_samples_data(
                        pa_matrix=pa_matrix, samples_to_include=samples_to_include
                    )
                    samples_pa_matrix.to_csv(samples_data_path.replace(".csv", "_unprocessed.csv"))
                    samples_pa_matrix.to_csv(samples_data_path, header=False)

            orig_dir = os.getcwd()
            os.chdir(output_dir)
            gemma_cmd = f"gemma -g {samples_data_path} -p {samples_trait_path} -k {kinship_matrix_path} -lmm 4 -o {test_results_suffix}"
            res = os.system(gemma_cmd)
            if res != 0:
                error_msg = (
                    f"test association failed due to error and so output was not created in {results_path}. exist "
                    f"code = {res} "
                )
                logger.error(error_msg)
                raise ValueError(error_msg)

            results = pd.read_csv(results_path, sep="\t")
            results.sort_values("p_wald", inplace=True)
            _, correct_p_wald, _, _ = multipletests(
                pvals=results.p_wald, alpha=0.05, method=multiple_test_correction_method, is_sorted=True,
            )  # add correction for multiple testing
            results["corrected_p_wald"] = correct_p_wald
            results.to_csv(results_path)
            os.chdir(orig_dir)
        else:
            results = pd.read_csv(res)
        return results
