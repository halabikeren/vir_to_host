import logging
import os

logger = logging.getLogger(__name__)
from enum import Enum
import typing as t
from collections import defaultdict

import pandas as pd
import numpy as np
import re

from Bio import Entrez, SeqIO

Entrez.email = "halabikeren@mail.tau.ac.il"
from habanero import Crossref

from functools import partial
import signal

from utils.signal_handling_service import SignalHandlingService


class DataCleanupUtils:
    @staticmethod
    def handle_duplicated_columns(colname: str, df: pd.DataFrame) -> pd.DataFrame:
        """
        :param colname: name of the column that was duplicated as a result of the merge
        :param df: dataframe to remove duplicated columns from
        :return:
        """
        duplicated_columns = [
            col for col in df.columns if colname in col and col != colname
        ]
        if len(duplicated_columns) == 0:
            return df
        main_colname = duplicated_columns[0]
        for col in duplicated_columns[1:]:

            # check that there are no contradictions
            contradictions = df.dropna(
                subset=[main_colname, col], how="any", inplace=False
            )
            contradictions = contradictions.loc[
                contradictions[main_colname] != contradictions[col]
            ]
            if contradictions.shape[0] > 0:
                logger.error(
                    f"{contradictions.shape[0]} contradictions found between column values in {main_colname} and {col}. original column values will be overridden"
                )
                df.loc[df.index.isin(contradictions.index), main_colname] = df.loc[
                    df.index.isin(contradictions.index), main_colname
                ].apply(
                    lambda x: contradictions.loc[
                        contradictions[main_colname] == x, col
                    ].values[0]
                )
            df[main_colname] = df[main_colname].fillna(df[col])
        df = df.rename(columns={main_colname: colname})
        for col in duplicated_columns[1:]:
            df = df.drop(col, axis="columns")
        return df


class RefSource(Enum):
    PAPER_DETAILS = 1
    SEQ_ID = 2
    GENE_ID = 3
    PUBMED_ID = 4
    OTHER = 5


class ReferenceCollectingUtils:
    @staticmethod
    def get_references(
        record: pd.Series, references_field: str, ref_to_doi: t.Dict[str, list]
    ) -> t.Optional[str]:
        """
        :param record: data record in the form od a pandas series
        :param references_field: name of the column which holds references data
        :param ref_to_doi: dictionary mapping items in the references field to dois
        :return:
        """
        references = record[references_field]
        if type(references) is float and np.isnan(references):
            return references
        if type(references) is str:
            references = [references]
        dois_united = []
        for reference in references:
            if reference in ref_to_doi:
                dois_united += ref_to_doi[reference]
        return ",".join(dois_united)

    @staticmethod
    def collect_dois(
        df: pd.DataFrame,
        output_field_name: str,
        references_field: str,
        source_type: RefSource,
        output_path: str,
    ):
        """
        :param df: dataframe to add a references by DOIs columns to
        :param output_field_name: the name of the added field
        :param references_field: a list of fields by which be extracted
        :param source_type: the type of query to conduct to ge the doi
        :param output_path: path to write temporary output to
        :return: none
        """

        # set signal handling
        signal.signal(
            signal.SIGINT, partial(SignalHandlingService.exit_handler, df, output_path),
        )
        signal.signal(
            signal.SIGTERM,
            partial(SignalHandlingService.exit_handler, df, output_path),
        )

        if source_type != RefSource.PAPER_DETAILS:
            df.loc[df[references_field].notnull(), references_field] = df.loc[
                df[references_field].notnull(), references_field
            ].apply(lambda x: re.split(",|;", str(x)))
        num_records = 0
        for chunk in np.array_split(df, (len(df.index) + 2) / 42):
            num_records += chunk.shape[0]
            if source_type != RefSource.PAPER_DETAILS:
                references = set(
                    [y for x in chunk[references_field].dropna() for y in x]
                )
            else:
                references = set([x for x in chunk[references_field].dropna()])
            if len(references) == 0:
                continue
            ref_to_doi = defaultdict(list)
            refs_query = ",".join(references)
            if source_type in [
                RefSource.SEQ_ID,
                RefSource.GENE_ID,
                RefSource.PUBMED_ID,
            ]:
                try:
                    db = (
                        "pubmed" if source_type == RefSource.PUBMED_ID else "nucleotide"
                    )
                    getter = (
                        Entrez.esummary
                        if source_type == RefSource.PUBMED_ID
                        else Entrez.efetch
                    )
                    matches = [
                        record
                        for record in Entrez.read(
                            getter(
                                db=db,
                                id=refs_query,
                                retmode="xml",
                                retmax=len(references),
                            )
                        )
                    ]
                    for match in matches:
                        doi = []
                        if source_type == RefSource.PUBMED_ID and "DOI" in match:
                            doi = [match["DOI"]]
                        elif source_type != RefSource.PUBMED_ID:
                            for ref in match["GBSeq_references"]:
                                if (
                                    "GBReference_xref" in ref
                                    and "GBXref_dbname" in ref["GBReference_xref"][0]
                                    and ref["GBReference_xref"][0]["GBXref_dbname"]
                                    == "doi"
                                ):
                                    doi.append(ref["GBReference_xref"][0]["GBXref_id"])
                        key = (
                            match["Id"]
                            if source_type == RefSource.PUBMED_ID
                            else (
                                match["GBSeq_accession-version"]
                                if source_type == RefSource.SEQ_ID
                                else [
                                    s.split("|")[-1]
                                    for s in match["GBSeq_other-seqids"]
                                    if "gi|" in s
                                ][0]
                            )
                        )
                        for item in doi:
                            ref_to_doi[key].append(item)
                except Exception as e:
                    logger.error(
                        f"failed to extract doi from references {refs_query} by {source_type.name} due to error {e}"
                    )
            elif source_type == RefSource.PAPER_DETAILS:
                cr = Crossref()
                for ref in references:
                    try:
                        res = cr.works(
                            query_bibliographic=ref, limit=1
                        )  # couldn't find a batch option
                        ref_to_doi[ref].append(res["message"]["items"][0]["DOI"])
                    except Exception as e:
                        logger.error(
                            f"failed to extract DOI for ref {ref} based on {source_type.name} due to error {e}"
                        )
            else:
                logger.error(
                    f"No mechanism is available for extraction of DOI from source type {source_type.name}"
                )

            df.loc[
                (df.index.isin(chunk.index)) & (df[references_field].notnull()),
                output_field_name,
            ] = df.loc[
                (df.index.isin(chunk.index)) & (df[references_field].notnull())
            ].apply(
                func=lambda x: ReferenceCollectingUtils.get_references(
                    x, references_field, ref_to_doi
                ),
            )
            df.to_csv(output_path, ignore_index=True)
            logger.info(f"Processed DOI data for {num_records} records")

    @staticmethod
    def unite_references(association_record: pd.Series) -> str:
        """
        :param association_record: pandas series corresponding to all columns holding reference data in the form of DOIs
        :return: string representing all the unique DOIs
        """
        references = set()
        for value in association_record.unique():
            if type(value) is list:
                references.add(value.split(","))
            elif type(value) is str:
                references.add(value)
        return ",".join(list(references))


class TaxonomyCollectingUtils:
    @staticmethod
    def collect_taxonomy_data(
        df: pd.DataFrame, taxonomy_data_dir: str,
    ) -> pd.DataFrame:
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
            signal.SIGTERM,
            partial(SignalHandlingService.exit_handler, df, output_path),
        )

        df = df.applymap(
            lambda s: s.lower() if isinstance(s, str) else s
        )  # make all strings lowercase to account for inconsistency between databases

        logger.info("complementing missing virus and host taxon ids from names.dmp")
        taxonomy_names_df = pd.read_csv(
            f"{taxonomy_data_dir}/names.dmp",
            sep="|",
            header=None,
            index_col=False,
            names=["tax_id", "name_txt", "unique_name", "class_name"],
        )
        taxonomy_names_df.replace(to_replace="\t", value="", regex=True, inplace=True)
        taxonomy_names_df.replace(to_replace="", value=np.nan, regex=True, inplace=True)
        taxonomy_names_df = taxonomy_names_df.applymap(
            lambda s: s.lower() if isinstance(s, str) else s
        )
        logger.info(
            f"#missing virus taxon ids before addition= {df.loc[df.virus_taxon_id.isna()].shape[0]}"
        )
        virus_taxon_names_df = taxonomy_names_df.loc[
            taxonomy_names_df.name_txt.isin(df.virus_taxon_name.unique())
        ][["tax_id", "name_txt"]]
        df.set_index(["virus_taxon_name"], inplace=True)
        df["virus_taxon_id"].fillna(
            value=virus_taxon_names_df.set_index("name_txt")["tax_id"].to_dict(),
            inplace=True,
        )
        df.reset_index(inplace=True)
        logger.info(
            (
                f"#missing virus taxon ids after addition = {df.loc[df.virus_taxon_id.isna()].shape[0]}"
            )
        )

        logger.info(
            (
                f"#missing host taxon ids before addition = {df.loc[df.host_taxon_id.isna()].shape[0]}"
            )
        )
        host_taxon_names_df = taxonomy_names_df.loc[
            taxonomy_names_df.name_txt.isin(df.host_taxon_name.unique())
        ][["tax_id", "name_txt"]]
        df.set_index(["host_taxon_name"], inplace=True)
        df["host_taxon_id"].fillna(
            value=host_taxon_names_df.set_index("name_txt")["tax_id"].to_dict(),
            inplace=True,
        )
        df.reset_index(inplace=True)
        logger.info(
            (
                f"#missing host taxon ids after addition = {df.loc[df.host_taxon_id.isna()].shape[0]}"
            )
        )
        df.reset_index(inplace=True)

        # fill in virus and host lineage info
        logger.info(
            "complementing missing virus and host taxon lineage info from rankedlineage.dmp"
        )
        logger.info(f"# missing cells before addition:\n {df.isnull().sum()}")
        logger.info(f"# missing cells:\n {df.isnull().sum()}")
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
        taxonomy_lineage_df.replace(
            to_replace="", value=np.nan, regex=True, inplace=True
        )
        taxonomy_lineage_df = taxonomy_lineage_df.applymap(
            lambda s: s.lower() if isinstance(s, str) else s
        )

        virus_taxonomy_lineage_df = taxonomy_lineage_df.loc[
            taxonomy_lineage_df.tax_name.isin(df.virus_taxon_name.unique())
        ].rename(
            columns={
                col: f"virus_{col.replace('tax_id', 'taxon').replace('tax_name', 'taxon')}_{'id' if 'id' in col else 'name'}"
                for col in taxonomy_lineage_df.columns
            },
        )
        df.set_index(["virus_taxon_name"], inplace=True)
        virus_taxonomy_lineage_df.set_index(["virus_taxon_name"], inplace=True)
        for col in virus_taxonomy_lineage_df.columns:
            if col not in df.columns and col != "virus_taxon_name":
                df[col] = np.nan
            values = virus_taxonomy_lineage_df[col].to_dict()
            df[col].fillna(value=values, inplace=True)
        df.reset_index(inplace=True)
        df.set_index(["virus_species_name"], inplace=True)
        for col in virus_taxonomy_lineage_df.columns:
            if col in [
                c
                for c in virus_taxonomy_lineage_df.columns
                if c != "virus_species_name"
            ]:
                if col not in df.columns:
                    df[col] = np.nan
                values = virus_taxonomy_lineage_df[col].to_dict()
                df[col].fillna(value=values, inplace=True)
        df.reset_index(inplace=True)

        host_taxonomy_lineage_df = taxonomy_lineage_df.loc[
            taxonomy_lineage_df.tax_name.isin(df.host_taxon_name.unique())
        ].rename(
            columns={
                col: f"host_{col.replace('tax_id', 'taxon').replace('tax_name', 'taxon')}_{'id' if 'id' in col else 'name'}"
                for col in taxonomy_lineage_df.columns
            },
        )
        df.set_index(["host_taxon_name"], inplace=True)
        host_taxonomy_lineage_df.set_index(["host_taxon_name"], inplace=True)
        for col in host_taxonomy_lineage_df.columns:
            if col in [
                c for c in host_taxonomy_lineage_df.columns if c != "host_taxon_name"
            ]:
                if col not in df:
                    df[col] = np.nan
                values = host_taxonomy_lineage_df[col].to_dict()
                df[col].fillna(value=values, inplace=True)
        df.reset_index(inplace=True)

        # fill missing taxon ids and their lineages using a more relaxed condition
        def find_taxon_id(taxon_name, taxonomy_df, field_prefix):
            match = taxonomy_df.loc[
                (
                    taxonomy_df[f"{field_prefix}_taxon_name"].str.contains(
                        f"{taxon_name}/", case=False
                    )
                )
                | (
                    taxonomy_df[f"{field_prefix}_taxon_name"].str.contains(
                        f"/{taxon_name}", case=False
                    )
                ),
                f"{field_prefix}_taxon_id",
            ]
            if match.shape[0] > 0:
                return match.values[0]
            return np.nan

        virus_taxonomy_lineage_df.reset_index(inplace=True)
        df.loc[df["virus_taxon_id"].isna(), "virus_taxon_id"] = df.loc[
            df["virus_taxon_id"].isna(), "virus_taxon_name"
        ].apply(
            lambda x: find_taxon_id(
                x, taxonomy_df=virus_taxonomy_lineage_df, field_prefix="virus"
            )
        )

        df.loc[df["virus_taxon_id"].isna(), "virus_taxon_id"] = df.loc[
            df["virus_taxon_id"].isna(), "virus_taxon_name"
        ].apply(
            lambda x: find_taxon_id(
                x, taxonomy_df=virus_taxonomy_lineage_df, field_prefix="virus"
            )
        )

        df.set_index(["virus_taxon_id"], inplace=True)
        virus_taxonomy_lineage_df.set_index(["virus_taxon_id"], inplace=True)
        for col in virus_taxonomy_lineage_df.columns:
            if col in [
                c for c in virus_taxonomy_lineage_df.columns if c != "virus_taxon_id"
            ]:
                values = virus_taxonomy_lineage_df[col].to_dict()
                df[col].fillna(value=values, inplace=True)
        df.reset_index(inplace=True)

        host_taxonomy_lineage_df.reset_index(inplace=True)
        df.loc[df["host_taxon_id"].isna(), "host_taxon_id"] = df.loc[
            df["host_taxon_id"].isna(), "host_taxon_name"
        ].apply(
            lambda x: find_taxon_id(
                x, taxonomy_df=host_taxonomy_lineage_df, field_prefix="host"
            )
        )

        df.set_index(["host_taxon_id"], inplace=True)
        host_taxonomy_lineage_df.set_index(["host_taxon_id"], inplace=True)
        for col in [
            c for c in host_taxonomy_lineage_df.columns if c != "host_taxon_id"
        ]:
            values = host_taxonomy_lineage_df[col].to_dict()
            df[col].fillna(value=values, inplace=True)
        df.reset_index(inplace=True)

        host_taxonomy_lineage_df["host_is_mammalian"] = host_taxonomy_lineage_df[
            "host_class_name"
        ].apply(lambda x: 1 if x == "mammalia" else 0)

        logger.info(f"# missing cells after addition:\n {df.isnull().sum()}")

        # fill rank of virus and host taxa
        logger.info("extracting rank of virus and host taxa")
        logger.info(f"# missing cells before addition= {df.isnull().sum()}")
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
        taxonomy_ranks_df = taxonomy_ranks_df.applymap(
            lambda s: s.lower() if isinstance(s, str) else s
        )
        taxonomy_ranks_df.replace(to_replace="\t", value="", regex=True, inplace=True)
        taxonomy_ranks_df.replace(to_replace="", value=np.nan, regex=True, inplace=True)
        virus_rank_df = taxonomy_ranks_df.loc[
            taxonomy_ranks_df.tax_id.isin(df.virus_taxon_id.unique())
        ]
        df["virus_taxon_rank"] = np.nan
        df.set_index(["virus_taxon_id"], inplace=True)
        values = virus_rank_df.set_index("tax_id")["rank"].to_dict()
        df["virus_taxon_rank"].fillna(value=values, inplace=True)
        df.reset_index(inplace=True)

        host_rank_df = taxonomy_ranks_df.loc[
            taxonomy_ranks_df.tax_id.isin(df.host_taxon_id.unique())
        ]
        df["host_taxon_rank"] = np.nan
        df.set_index(["host_taxon_id"], inplace=True)
        values = host_rank_df.set_index("tax_id")["rank"].to_dict()
        df["host_taxon_rank"].fillna(value=values, inplace=True)
        df.reset_index(inplace=True)

        df.loc[df.virus_strain_name.notnull(), "virus_is_species"] = 0
        df.loc[df.virus_strain_name.isnull(), "virus_is_species"] = 1
        df.loc[
            (df.virus_species_name.isnull()) & (df.virus_is_species == 1),
            "virus_species_name",
        ] = df.loc[
            (df.virus_species_name.isnull()) & (df.virus_is_species == 1),
            "virus_taxon_name",
        ]
        df.loc[
            (df.host_species_name.isnull()) & (df.host_taxon_rank == "species"),
            "host_species_name",
        ] = df.loc[
            (df.host_species_name.isnull()) & (df.host_taxon_rank == "species"),
            "host_taxon_name",
        ]
        df.loc[df.virus_strain_name.notnull(), "virus_taxon_rank"] = "strain"
        df = df[
            [col for col in df if "_id" not in col and col != "index"]
            + ["virus_taxon_id", "host_taxon_id"]
        ]
        df.loc[
            (df.host_is_mammalian.isna()) & (df.host_class_name.notna()),
            "host_is_mammalian",
        ] = df.loc[
            (df.host_is_mammalian.isna()) & (df.host_class_name.notna()),
            "host_class_name",
        ].apply(
            lambda x: 1 if x == "mammalia" else 0
        )

        if "virus_species_id" not in df.columns:
            df["virus_species_id"] = np.nan
        df.loc[
            (df["virus_species_id"].isna()) & (df["virus_taxon_rank"] == "species"),
            "virus_species_id",
        ] = df.loc[
            (df["virus_species_id"].isna()) & (df["virus_taxon_rank"] == "species"),
            "virus_taxon_id",
        ]

        virus_taxonomy_lineage_df.reset_index(inplace=True)
        df.loc[
            (df["virus_species_id"].isna())
            & (df["virus_taxon_rank"] == "species")
            & (df["virus_species_name"].notna()),
            "virus_species_id",
        ] = df.loc[
            (df["virus_species_id"].isna())
            & (df["virus_taxon_rank"] == "species")
            & (df["virus_species_name"].notna()),
            "virus_species_name",
        ].apply(
            lambda x: find_taxon_id(
                x, taxonomy_df=virus_taxonomy_lineage_df, field_prefix="virus"
            )
        )

        logger.info(f"# missing cells after addition= {df.isnull().sum()}")

        return df


class SequenceCollectingUtils:
    @staticmethod
    def get_sequence(record: str, sequence_data):
        regex = re.compile("(\w*_\s*\d*)")
        accessions = [item.group(1).replace(" ", "") for item in regex.finditer(record)]
        seq = ""
        for acc in accessions:
            if acc in sequence_data:
                seq += str(sequence_data[acc])
            else:
                print(f"record={record}, missing acc={acc}")
                return np.nan
        if len(seq) == 0:
            print(f"record={record}")
            return np.nan
        return seq

    @staticmethod
    def get_accession(record: str):
        regex = re.compile("(\w*_\s*\d*)")
        accessions = [
            item.group(1)
            .replace(" ", "")
            .replace("L: ", "")
            .replace("M: ", "")
            .replace("S: ", "")
            .replace("; ", ",")
            for item in regex.finditer(record)
        ]
        return ",".join(accessions)

    @staticmethod
    def extract_accessions(accession_records):
        regex = re.compile("(\w*\s*\d*)")
        accessions = []
        for record in accession_records:
            record_accessions = [
                item.group(1).replace(" ", "") for item in regex.finditer(record)
            ]
            accessions += record_accessions
        return accessions

    @staticmethod
    def get_seq_data_from_virus_name(name, seq_data):
        acc, seq = np.nan, np.nan
        if name.values[0] in seq_data:
            data = seq_data[name.values[0]]
            acc = data[0]
            seq = data[1]
        return acc, seq

    @staticmethod
    def get_sequence_info(path: str) -> t.Tuple[t.Dict[str, str], t.Dict[str, str]]:
        sequence_data = list(SeqIO.parse(path, format="fasta"))
        name_regex = re.compile(
            "gb\:([^|]*)\|Organism\:([^|]*)\|.*?Strain Name\:([^|]*)"
        )
        virus_taxon_name_to_seq = {
            name_regex.search(item.description).group(2).lower()
            + " "
            + name_regex.search(item.description).group(3).lower(): str(item.seq)
            for item in sequence_data
        }
        virus_taxon_name_to_gb = {
            name_regex.search(item.description).group(2).lower()
            + " "
            + name_regex.search(item.description)
            .group(3)
            .lower(): name_regex.search(item.description)
            .group(1)
            for item in sequence_data
        }
        return virus_taxon_name_to_seq, virus_taxon_name_to_gb
