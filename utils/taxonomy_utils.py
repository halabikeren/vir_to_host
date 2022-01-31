import os
import signal
import logging
from functools import partial
from time import sleep
from pygbif import species
import numpy as np
import pandas as pd
from Bio import Entrez

logger = logging.getLogger(__name__)

from .signal_handling_service import SignalHandlingService


class TaxonomyCollectingUtils:
    @staticmethod
    def collect_taxonomy_data_from_gbif_api(df: pd.DataFrame, data_prefix: str) -> pd.DataFrame:
        """
        :param df: dataframe with taxonomy data to fill
        :param data_prefix: either virus or host
        :return: dataframe with taxa data, complemented from ncbi api
        """
        gbif_lineage_keys_to_ncbi_keys = {
            "kingdom": f"{data_prefix}_kingdom_name",
            "phylum": f"{data_prefix}_phylum_name",
            "order": f"{data_prefix}_order_name",
            "family": f"{data_prefix}_family_name",
            "genus": f"{data_prefix}_genus_name",
            "species": f"{data_prefix}_species_name",
            "rank": f"{data_prefix}_taxon_rank",
        }

        record_names_with_missing_data = [
            name.rstrip()
            for name in list(df.loc[(df[f"{data_prefix}_taxon_id"].isna()), f"{data_prefix}_taxon_name"].unique())
        ]

        logger.info(
            f"extracting {data_prefix} data from gbif for {len(record_names_with_missing_data)} records: both lineage data and correction of taxon name for consecutive search in ncbi"
        )

        gbif_data = {name: species.name_suggest(q=name) for name in record_names_with_missing_data}
        gbif_relevant_data = {name: gbif_data[name][0] for name in gbif_data if len(gbif_data[name]) > 0}

        logger.info(f"completed extraction from api, found {len(gbif_relevant_data.keys())} relevant records")

        # complete missing data from gbif
        df.set_index(f"{data_prefix}_taxon_name", inplace=True)
        for key in gbif_lineage_keys_to_ncbi_keys:
            gbif_key_data = {
                name: gbif_relevant_data[name][key].lower()
                for name in gbif_relevant_data
                if key in gbif_relevant_data[name]
            }
            df[gbif_lineage_keys_to_ncbi_keys[key]].fillna(value=gbif_key_data, inplace=True)
        df.reset_index(inplace=True)

        # correct taxon names to scientific names
        avail_name_to_scientific_name = {
            name: gbif_relevant_data[name]["scientificName"].lower() for name in gbif_relevant_data
        }
        df[f"{data_prefix}_taxon_name"] = df[f"{data_prefix}_taxon_name"].replace(avail_name_to_scientific_name)

        return df

    @staticmethod
    def collect_taxonomy_data_from_ncbi_api(df: pd.DataFrame, data_prefix: str) -> pd.DataFrame:
        """
        :param df: dataframe with taxonomy data to fill
        :param data_prefix: either virus or host
        :return: dataframe with taxa data, complemented from ncbi api
        """
        if f"{data_prefix}_species_id" not in df.columns:
            df[f"{data_prefix}_species_id"] = np.nan
        species_names = list(df.loc[df[f"{data_prefix}_species_id"].isna(), f"{data_prefix}_species_name"].unique())

        logger.info(f"extracting {data_prefix} data from ncbi for {len(species_names)} records: tax ids")

        species_name_to_data = dict()
        i = 0
        while i < len(species_names):
            name = species_names[i]
            try:
                species_name_to_data[name] = Entrez.read(Entrez.esearch(db="taxonomy", term=name, retmode="xml"))
                i += 1
            except Exception as e:
                if "429" in str(e):
                    sleep(2)
                else:
                    continue
        species_name_to_id = {
            name: species_name_to_data[name]["IdList"][0]
            for name in species_name_to_data
            if "IdList" in species_name_to_data[name] and len(species_name_to_data[name]["IdList"]) > 0
        }

        logger.info(f"extraction is complete. found {len(species_name_to_id.keys())} relevant records")

        df.set_index(f"{data_prefix}_species_name", inplace=True)
        if f"{data_prefix}_species_id" not in df.columns:
            df[f"{data_prefix}_species_id"] = np.nan
        df[f"{data_prefix}_species_id"].fillna(value=species_name_to_id, inplace=True)
        df.reset_index(inplace=True)
        return df

    @staticmethod
    def collect_tax_ids(df: pd.DataFrame, taxonomy_names_df: pd.DataFrame, data_prefix: str) -> pd.DataFrame:
        """
        :param df: dataframe with taxonomy data to fill
        :param taxonomy_names_df: dataframe with taxa names and ids
        :param data_prefix: either virus or host
        :return: dataframe with taxa ids
        """
        logger.info(f"complementing missing {data_prefix} taxon ids from names.dmp")
        logger.info(
            (
                f"#missing {data_prefix} taxon ids before addition = {df.loc[df[f'{data_prefix}_taxon_id'].isna()].shape[0]}"
            )
        )

        taxon_names_df = taxonomy_names_df.loc[
            taxonomy_names_df.name_txt.isin(
                df.loc[df[f"{data_prefix}_taxon_id"].isna(), f"{data_prefix}_taxon_name"].unique()
            )
        ][["tax_id", "name_txt"]]
        tax_name_to_id = taxon_names_df.set_index("name_txt")["tax_id"].to_dict()
        df.set_index([f"{data_prefix}_taxon_name"], inplace=True)
        df["virus_taxon_id"].fillna(
            value=tax_name_to_id, inplace=True,
        )
        df.reset_index(inplace=True)

        logger.info(
            (
                f"#missing {data_prefix} taxon ids after addition = {df.loc[df[f'{data_prefix}_taxon_id'].isna()].shape[0]}"
            )
        )

        return df

    @staticmethod
    def collect_lineage_info(df: pd.DataFrame, taxonomy_lineage_df: pd.DataFrame, data_prefix: str) -> pd.DataFrame:
        """
        :param df: dataframe with taxonomy data to fill
        :param taxonomy_lineage_df: dataframe with taxonomy lineage data
        :param data_prefix: either virus or host
        :return: dataframe with taxa lineage info
        """

        logger.info(f"complementing missing {data_prefix} taxon lineage info from rankedlineage.dmp")

        taxonomy_lineage_df = taxonomy_lineage_df.loc[
            taxonomy_lineage_df.tax_name.isin(df.virus_taxon_name.unique())
        ].rename(
            columns={
                col: f"{data_prefix}_{col.replace('tax_id', 'taxon').replace('tax_name', 'taxon')}_{'id' if 'id' in col else 'name'}"
                for col in taxonomy_lineage_df.columns
            },
        )

        # fill one by taxon name, regardless of rank, and once by species rank, for which more lineage data is available
        fill_by_fields = {
            f"{data_prefix}_taxon_name": "tax_name",
            f"{data_prefix}_taxon_id": "tax_id",
            f"{data_prefix}_species_name": "species",
        }
        for field in fill_by_fields.keys():
            if field in df.columns and fill_by_fields[field] in taxonomy_lineage_df.columns:
                relevant_df = df.loc[df[field].notna()]
                relevant_df.set_index(field, inplace=True)
                taxonomy_lineage_df.set_index(fill_by_fields[field], inplace=True)
                for col in taxonomy_lineage_df.columns:
                    if col not in df.columns and col != field:
                        df[col] = np.nan
                    if col not in relevant_df.columns and col != field:
                        relevant_df[col] = np.nan
                    values = taxonomy_lineage_df[col].to_dict()
                    relevant_df[col].fillna(value=values, inplace=True)
                relevant_df.to_csv(f"{os.getcwd()}taxonomy_lineage_df_missing_values.csv", index=False)
                df.update(relevant_df[~relevant_df.index.duplicated()])
                relevant_df.reset_index(inplace=True)
                taxonomy_lineage_df.reset_index(inplace=True)

        df.reset_index(inplace=True)

        return df

    @staticmethod
    def collect_tax_rank(df: pd.DataFrame, taxonomy_ranks_df: pd.DataFrame, data_prefix: str) -> pd.DataFrame:
        """
        :param df: dataframe with taxonomy data to fill
        :param taxonomy_ranks_df: dataframe with ranks of taxa
        :param data_prefix: either virus or host
        :return: dataframe with taxa rank info
        """

        logger.info(f"extracting rank of {data_prefix}")
        rank_df = taxonomy_ranks_df.loc[taxonomy_ranks_df.tax_id.isin(df[f"{data_prefix}_taxon_id"].unique())]
        df[f"{data_prefix}_taxon_rank"] = np.nan
        df.set_index([f"{data_prefix}_taxon_id"], inplace=True)
        values = rank_df.set_index("tax_id")["rank"].to_dict()
        df[f"{data_prefix}_taxon_rank"].fillna(value=values, inplace=True)
        df.reset_index(inplace=True)

        if data_prefix == "virus":
            df.loc[df.virus_strain_name.notnull(), "virus_is_species"] = 0
            df.loc[df.virus_strain_name.isnull(), "virus_is_species"] = 1
            df.loc[(df.virus_species_name.isnull()) & (df.virus_is_species == 1), "virus_species_name",] = df.loc[
                (df.virus_species_name.isnull()) & (df.virus_is_species == 1), "virus_taxon_name",
            ]
            df.loc[df.virus_strain_name.notnull(), "virus_taxon_rank"] = "strain"
        else:
            df.loc[(df.host_species_name.isnull()) & (df.host_taxon_rank == "species"), "host_species_name",] = df.loc[
                (df.host_species_name.isnull()) & (df.host_taxon_rank == "species"), "host_taxon_name",
            ]
            df.loc[(df.host_is_mammalian.isna()) & (df.host_class_name.notna()), "host_is_mammalian",] = df.loc[
                (df.host_is_mammalian.isna()) & (df.host_class_name.notna()), "host_class_name",
            ].apply(lambda x: 1 if x == "mammalia" else 0)

        if f"{data_prefix}_species_id" not in df.columns:
            df[f"{data_prefix}_species_id"] = np.nan
        df.loc[
            (df[f"{data_prefix}_species_id"].isna()) & (df[f"{data_prefix}_taxon_rank"] == "species"),
            f"{data_prefix}_species_id",
        ] = df.loc[
            (df[f"{data_prefix}_species_id"].isna()) & (df[f"{data_prefix}_taxon_rank"] == "species"),
            f"{data_prefix}_taxon_id",
        ]

        return df

    @staticmethod
    def collect_taxonomy_data(df: pd.DataFrame, taxonomy_data_dir: str,) -> pd.DataFrame:
        """
        :param df: dataframe holding taxon names (and possibly ids) by which taxon data should be extracted
        :param taxonomy_data_dir: directory holding dump files of the NCBI taxonomy FTP services https://ftp.ncbi.nih.gov/pub/taxonomy/new_taxdump/
        :return: the processed dataframe
        """

        output_path = f"{os.getcwd()}/collect_taxonomy_data.csv"

        # set signal handling
        signal.signal(
            signal.SIGINT, partial(SignalHandlingService.exit_handler, df, output_path),
        )
        signal.signal(
            signal.SIGTERM, partial(SignalHandlingService.exit_handler, df, output_path),
        )

        # make all strings lowercase to account for inconsistency between databases
        df = df.applymap(lambda s: s.lower() if isinstance(s, str) else s)

        logger.info(f"# missing data before taxonomy data collection from ftp = {df.isnull().sum()}")

        # collect tax ids
        taxonomy_names_df = pd.read_csv(
            f"{taxonomy_data_dir}/names.dmp",
            sep="|",
            header=None,
            index_col=False,
            names=["tax_id", "name_txt", "unique_name", "class_name"],
        )
        taxonomy_names_df.replace(to_replace="\t", value="", regex=True, inplace=True)
        taxonomy_names_df.replace(to_replace="", value=np.nan, regex=True, inplace=True)
        taxonomy_names_df = taxonomy_names_df.applymap(lambda s: s.lower() if isinstance(s, str) else s)
        df = TaxonomyCollectingUtils.collect_tax_ids(df=df, taxonomy_names_df=taxonomy_names_df, data_prefix="virus")
        df = TaxonomyCollectingUtils.collect_tax_ids(df=df, taxonomy_names_df=taxonomy_names_df, data_prefix="host")

        # collect missing tax ids (and lineage info, if available) data using api requests to gbif and entrez
        # df = ParallelizationService.parallelize(df=df, func=partial(
        #     TaxonomyCollectingUtils.collect_taxonomy_data_from_gbif_api, data_prefix="virus"),
        #                                         num_of_processes=multiprocessing.cpu_count())
        df = TaxonomyCollectingUtils.collect_taxonomy_data_from_gbif_api(df=df, data_prefix="virus")
        # df = ParallelizationService.parallelize(df=df, func=partial(
        #     TaxonomyCollectingUtils.collect_taxonomy_data_from_gbif_api, data_prefix="host"),
        #                                         num_of_processes=2)  # multiprocessing.cpu_count())
        df = TaxonomyCollectingUtils.collect_taxonomy_data_from_gbif_api(df=df, data_prefix="host")

        # collect tax ids gain from ncbi to account for corrected tax names
        df = TaxonomyCollectingUtils.collect_tax_ids(df=df, taxonomy_names_df=taxonomy_names_df, data_prefix="virus")
        df = TaxonomyCollectingUtils.collect_tax_ids(df=df, taxonomy_names_df=taxonomy_names_df, data_prefix="host")
        # df = ParallelizationService.parallelize(df=df, func=partial(
        #     TaxonomyCollectingUtils.collect_taxonomy_data_from_ncbi_api, data_prefix="virus"),
        #                                         num_of_processes=2)  # multiprocessing.cpu_count())
        df = TaxonomyCollectingUtils.collect_taxonomy_data_from_ncbi_api(df=df, data_prefix="virus")
        # df = ParallelizationService.parallelize(df=df, func=partial(
        #     TaxonomyCollectingUtils.collect_taxonomy_data_from_ncbi_api, data_prefix="host"),
        #                                         num_of_processes=2)  # multiprocessing.cpu_count())
        df = TaxonomyCollectingUtils.collect_taxonomy_data_from_ncbi_api(df=df, data_prefix="host")

        df.to_csv(output_path, index=False)

        # collect lineage info
        taxonomy_lineage_df = pd.read_csv(
            f"{taxonomy_data_dir}/rankedlineage.dmp",
            sep="|",
            header=None,
            index_col=False,
            names=[
                "tax_id",
                "tax_name",
                "species",
                "genus",
                "family",
                "order",
                "class",
                "phylum",
                "kingdom",
                "superkingdom",
            ],
            dtype={"tax_id": np.float64},
        )
        taxonomy_lineage_df.replace(to_replace="\t", value="", regex=True, inplace=True)
        taxonomy_lineage_df.replace(to_replace="", value=np.nan, regex=True, inplace=True)
        taxonomy_lineage_df = taxonomy_lineage_df.applymap(lambda s: s.lower() if isinstance(s, str) else s)
        df = TaxonomyCollectingUtils.collect_lineage_info(
            df=df, taxonomy_lineage_df=taxonomy_lineage_df, data_prefix="virus"
        )
        df = TaxonomyCollectingUtils.collect_lineage_info(
            df=df, taxonomy_lineage_df=taxonomy_lineage_df, data_prefix="host"
        )

        df.to_csv(output_path, index=False)

        # collect rank info
        taxonomy_ranks_df = pd.read_csv(
            f"{taxonomy_data_dir}/nodes.dmp",
            sep="|",
            header=None,
            index_col=False,
            names=[
                "tax_id",
                "parent_tax_id",
                "rank",
                "embl_code",
                "division_id",
                "inherited_div_flag",
                "genetic_code_id",
                "inherited_GC_flag",
                "mitochondrial_genetic_code_id",
                "inherited_MGC_flag",
                "GenBank_hidden_flag",
                "hidden_subtree_root_flag",
                "comments",
                "plastid_genetic_code_id",
                "inherited_PGC_flag",
                "specified_species",
                "hydrogenosome_genetic_code_id",
                "inherited_HGC_flag",
            ],
        )
        taxonomy_ranks_df = taxonomy_ranks_df.applymap(lambda s: s.lower() if isinstance(s, str) else s)
        taxonomy_ranks_df.replace(to_replace="\t", value="", regex=True, inplace=True)
        taxonomy_ranks_df.replace(to_replace="", value=np.nan, regex=True, inplace=True)
        df = TaxonomyCollectingUtils.collect_tax_rank(df=df, taxonomy_ranks_df=taxonomy_ranks_df, data_prefix="virus")
        df = TaxonomyCollectingUtils.collect_tax_rank(df=df, taxonomy_ranks_df=taxonomy_ranks_df, data_prefix="host")

        df.to_csv(output_path, index=False)

        logger.info(f"# missing data after taxonomy data collection from ftp = {df.isnull().sum()}")

        return df
