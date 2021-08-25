import logging
import os
import subprocess
import time
from random import random

import Bio
from Bio.Seq import Seq

logger = logging.getLogger(__name__)
from enum import Enum
import typing as t
from collections import defaultdict
from Levenshtein import distance as lev

import pandas as pd
import numpy as np
import re

from Bio import Entrez, SeqIO

Entrez.email = "halabikeren@mail.tau.ac.il"
from habanero import Crossref

from functools import partial
import signal
import psutil
import random

from utils.signal_handling_service import SignalHandlingService

NUCLEOTIDES = ["A", "C", "G", "T"]
STOP_CODONS = Bio.Data.CodonTable.standard_dna_table.stop_codons
CODONS = list(Bio.Data.CodonTable.standard_dna_table.forward_table.keys()) + STOP_CODONS
AMINO_ACIDS = set(Bio.Data.CodonTable.standard_dna_table.forward_table.values())


class SequenceType(Enum):
    GENOME = 1
    CDS = 2
    PROTEIN = 3


class DinucleotidePositionType(Enum):
    REGULAR = 1
    BRIDGE = 2
    NONBRIDGE = 3


class GenomeType(Enum):
    RNA = 0
    DNA = 1
    UNKNOWN = np.nan


class RefSource(Enum):
    PAPER_DETAILS = 1
    SEQ_ID = 2
    GENE_ID = 3
    PUBMED_ID = 4
    OTHER = 5


class ClusteringMethod(Enum):
    CDHIT = 1


class ClusteringUtils:

    @staticmethod
    def get_sequences_similarity(viruses_names: str, viral_seq_data: pd.DataFrame) -> float:
        """
        :param viruses_names: names of viruses, separated by comma
        :param viral_seq_data: dataframe holding sequences of ny viruses names
        :return: similarity measure between 0 and 1 for the sequences of thew viruses, if available, induced by the number of cd-hit clusters at threshold 0.8 for the set of respective sequences
        """
        viruses_names_lst = viruses_names.split(",")
        relevant_virus_seq_data = list(viral_seq_data.loc[viral_seq_data.virus_taxon_name.isin(viruses_names_lst)][
                                           [col for col in viral_seq_data.columns if "sequence" in col]].dropna(
            axis=1).values.flatten())
        if len(relevant_virus_seq_data) == 0:
            return np.nan
        elif len(relevant_virus_seq_data) == 1:
            return 1
        aux_dir = f"{os.getcwd()}/cdhit_aux/"
        os.makedirs(aux_dir, exist_ok=True)
        rand_id = random.random()
        cdhit_input_path = f"{aux_dir}/sequences_{time.time()}_{os.getpid()}_{rand_id}.fasta"
        with open(cdhit_input_path, "w") as infile:
            infile.write(
                "\n".join([f">S{i}\n{relevant_virus_seq_data[i]}" for i in range(len(relevant_virus_seq_data))]))
        cdhit_output_path = f"{aux_dir}/cdhit_group_out_{time.time()}_{os.getpid()}_{rand_id}"
        cmd = f"cd-hit-est -i {cdhit_input_path} -o {cdhit_output_path} -c 0.99 -n 5"
        process = subprocess.Popen(
            cmd, shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE
        )
        if len(process.stderr.read()) > 0:
            raise RuntimeError(
                f"CD-HIT failed to properly execute and provide an output file with error {process.stderr.read()} and output is {process.stdout.read()}")
        with open(f"{cdhit_output_path}.clstr", "r") as clusters_path:
            dist = (clusters_path.read().count(">Cluster")-1) / len(relevant_virus_seq_data)
            similarity = 1 - dist
        logger.info(f"similarity of sequences across viruses {viruses_names} is {similarity}")
        process = subprocess.Popen(f"rm -r {cdhit_input_path}", shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        if len(process.stderr.read()) > 0:
            raise RuntimeError(f"failed to remove {cdhit_input_path} with error {process.stderr.read()} and output is {process.stdout.read()}")
        process = subprocess.Popen(f"rm -r {cdhit_output_path}", shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        if len(process.stderr.read()) > 0:
            raise RuntimeError(f"failed to remove {cdhit_output_path} with error {process.stderr.read()} and output is {process.stdout.read()}")
        process = subprocess.Popen(f"rm -r {cdhit_output_path}.clstr", shell=True, stdout=subprocess.PIPE,
                                   stderr=subprocess.PIPE)
        if len(process.stderr.read()) > 0:
            raise RuntimeError(
                f"failed to remove {cdhit_output_path}.clstr with error {process.stderr.read()} and output is {process.stdout.read()}")

        #distances = []
        #for pair in itertools.permutations(relevant_virus_seq_data, 2):
        #    distances.append(ClusteringUtils.get_pairwise_alignment_distance(pair[0], pair[1]))
        #similarity = 1 - np.mean(distances)

        # print(f"viruses_names = {viruses_names}, len(relevant_virus_seq_data) = {len(relevant_virus_seq_data)}, similarity = {similarity}")

        return similarity

    @staticmethod
    def get_cdhit_clusters(
        elements: pd.DataFrame,
        id_colname: str,
        seq_colnames: t.List[str],
        homology_threshold: float = 0.99,
    ) -> t.Dict[t.Union[np.int64, str], np.int64]:
        """
        :param elements: elements to cluster using kmeans
        :param id_colname: column holding the id of the elements
        :param seq_colnames: names of columns holding the sequences of the elements
        :param homology_threshold: cdhit threshold in clustering
        :return: a list of element ids corresponding the the representatives of the cdhit clusters
        """

        aux_dir = f"{os.getcwd()}/cdhit_aux/"
        os.makedirs(aux_dir, exist_ok=True)

        elm_to_seq = dict()
        elm_to_fake_name = dict()
        fake_name_to_elm = dict()
        for index, row in elements.iterrows():
            elm = row[id_colname]
            seq = row[seq_colnames].dropna().values[0]
            elm_to_fake_name[elm] = f"S{index}"
            fake_name_to_elm[f"S{index}"] = elm
            elm_to_seq[elm] = seq

        cdhit_input_path = f"{aux_dir}/sequences.fasta"
        with open(cdhit_input_path, "w") as infile:
            infile.write(
                "\n".join(
                    [
                        f">{elm_to_fake_name[elm]}\n{elm_to_seq[elm]}"
                        for elm in elm_to_seq
                    ]
                )
            )

        cdhit_output_file = f"{aux_dir}/cdhit_out_thr_{homology_threshold}"
        if not os.path.exists(cdhit_output_file):
            word_len = (
                (5 if homology_threshold > 0.7 else 4)
                if homology_threshold > 0.6
                else (3 if homology_threshold > 0.5 else 2)
            )
            cmd = f"cd-hit-est -i {cdhit_input_path} -o {cdhit_output_file} -c {homology_threshold} -n {word_len}"
            process = subprocess.Popen(
                cmd, shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE
            )
            if len(process.stderr.read()) > 0:
                raise RuntimeError(
                    f"CD-HIT failed to properly execute and provide an output file with error {process.stderr.read()} and output is {process.stdout.read()}"
                )

        elm_to_cluster = dict()
        clusters_data_path = f"{cdhit_output_file}.clstr"
        member_regex = re.compile(">(.*?)\.\.\.", re.MULTILINE | re.DOTALL)
        with open(clusters_data_path, "r") as outfile:
            clusters = outfile.read().split(">Cluster")[1:]
            for cluster in clusters:
                data = cluster.split("\n")
                cluster_id = np.int64(data[0])
                cluster_members = []
                for member_data in data[1:]:
                    if len(member_data) > 0:
                        member_fake_name = member_regex.search(member_data).group(1)
                        member = fake_name_to_elm[member_fake_name]
                        cluster_members.append(member)
                elm_to_cluster.update(
                    {member: cluster_id for member in cluster_members}
                )

        return elm_to_cluster

    @staticmethod
    def compute_clusters_representatives(
        elements: pd.DataFrame,
        id_colname: str,
        seq_colnames: t.List[str],
        clustering_method: ClusteringMethod = ClusteringMethod.CDHIT,
        homology_threshold: t.Optional[float] = 0.99,
    ):
        """
        :param elements: elements to cluster using cdhit
        :param id_colname: column holding the id of the elements
        :param seq_colnames: names of columns holding the sequences of the elements
        :param clustering_method: either cdhit or kmeans
        :param homology_threshold: cdhit threshold in clustering
        :return: none, adds cluster_id and cluster_representative columns to the existing elements dataframe
        """
        if clustering_method == ClusteringMethod.CDHIT:
            elm_to_cluster = ClusteringUtils.get_cdhit_clusters(
                elements=elements,
                id_colname=id_colname,
                seq_colnames=seq_colnames,
                homology_threshold=homology_threshold,
            )
        else:
            logger.error(f"clustering method {clustering_method} is not implemented")
            raise ValueError(
                f"clustering method {clustering_method} is not implemented"
            )
        elements["cluster_id"] = np.nan
        elements.set_index(id_colname, inplace=True)
        elements["cluster_id"].fillna(value=elm_to_cluster, inplace=True)
        elements.reset_index(inplace=True)

        clusters = list(set(elm_to_cluster.values()))
        cluster_to_representative = dict()
        for cluster in clusters:
            cluster_members = elements.loc[elements.cluster_id == cluster]
            if cluster_members.shape[0] == 1:
                cluster_representative = cluster_members.iloc[0][id_colname]
            else:
                elements_distances = ClusteringUtils.compute_pairwise_sequence_distances(
                    elements=cluster_members,
                    id_colname=id_colname,
                    seq_colnames=seq_colnames,
                )
                cluster_representative = ClusteringUtils.get_centroid(
                    elements_distances
                )
            cluster_to_representative[cluster] = cluster_representative

        elements["cluster_representative"] = np.nan
        elements.set_index("cluster_id", inplace=True)
        elements["cluster_representative"].fillna(
            value=cluster_to_representative, inplace=True
        )
        elements.reset_index(inplace=True)

    @staticmethod
    def get_pairwise_alignment_distance(seq1: str, seq2: str) -> float:
        """
        :param seq1: sequence 1
        :param seq2: sequence 2
        :return: a float between 0 and 1 representing the distance between the two sequences based on their pairwise alignment
        """
        try:
            dist = lev(seq1, seq2) / np.max([len(seq1), len(seq2)])
            return dist
        except Exception as e:
            logger.error(f"failed to compute distance due to error: {e}")
            logger.error(f"len(seq1)={len(seq1)}, len(seq2)={len(seq2)}")
            process = psutil.Process(os.getpid())
            logger.error(process.memory_info().rss)  # in bytes
            return np.nan

    @staticmethod
    def get_distance(x, id_colname, seq_colnames, elements):
        elm1 = x["element_1"]
        elm2 = x["element_2"]
        elm1_seq = (
            elements.loc[elements[id_colname] == elm1][seq_colnames]
            .dropna(axis=1)
            .values[0][0]
        )
        elm2_seq = (
            elements.loc[elements[id_colname] == elm2][seq_colnames]
            .dropna(axis=1)
            .values[0][0]
        )
        return ClusteringUtils.get_pairwise_alignment_distance(elm1_seq, elm2_seq)

    @staticmethod
    def compute_pairwise_sequence_distances(
        elements: pd.DataFrame,
        id_colname: str,
        seq_colnames: t.List[str],
        distance_function: t.Callable[
            [str, str], float
        ] = get_pairwise_alignment_distance,
    ) -> pd.DataFrame:
        """
        :param elements: elements to compute pairwise distances for
        :param id_colname: column holding the id of the elements
        :param seq_colnames: names of columns holding the sequences of the elements
        :param distance_function: function that receives two sequences and returns a float 
        :return: a dataframe with row1 as element id, row 2 as element id and row3 ad the pairwise distance between the two elements correspond to ids in row1 and row2
        """
        elements_distances = pd.DataFrame(
            [
                (elm1, elm2)
                for elm1 in elements[id_colname]
                for elm2 in elements[id_colname]
            ],
            columns=["element_1", "element_2"],
        )

        elements_distances["distance"] = elements_distances.apply(
            lambda x: ClusteringUtils.get_distance(
                x, id_colname, seq_colnames, elements
            ),
            axis=1,
        )

        return elements_distances

    @staticmethod
    def get_centroid(elements_distances: pd.DataFrame) -> t.Union[np.int64, str]:
        """
        :param elements_distances: a dataframe with row1 as element id, row 2 as element id and row3 ad the pairwise distance between the two elements correspond to ids in row1 and row2
        :return: the element id of the centroid
        """
        elements_sum_distances = elements_distances.groupby("element_1")["distance"].sum().reset_index()
        centroid = elements_sum_distances.iloc[
            elements_distances["distance"].argmin()
        ]["element_1"]
        return centroid


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

        # collect missing species data using api requests
        species_names = list(df.loc[df["virus_species_id"].isna(), "virus_species_name"].unique())
        species_name_to_data = {name: Entrez.parse(Entrez.esearch(db="taxonomy", term=name, retmode="xml")) for name in
                   species_names}
        species_name_to_id = {name: species_name_to_data[name]['IdList'][0] for name in species_name_to_data if
                         'IdList' in species_name_to_data[name] and len(species_name_to_data[name]['IdList']) > 0}
        df.set_index("virus_species_name", inplace=True)
        df["virus_species_id"].fillna(value=species_name_to_id, inplace=True)
        df.reset_index(inplace=True)


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

    @staticmethod
    def extract_sequences_from_record(
        sequences: dict, record: t.List[t.Dict[str, str]], type: SequenceType
    ):
        if type == SequenceType.GENOME:
            sequences[type].append(record[0]["GBSeq_sequence"])
        else:
            coding_regions = [
                record[0]["GBSeq_feature-table"][i]
                for i in range(len(record[0]["GBSeq_feature-table"]))
                if record[0]["GBSeq_feature-table"][i]["GBFeature_key"] == "CDS"
            ]
            for coding_region in coding_regions:
                if type == SequenceType.CDS:
                    cds_start = int(
                        coding_region["GBInterval_from"][0]["GBInterval_from"]
                    )
                    cds_end = int(coding_region["GBInterval_from"][0]["GBInterval_to"])
                    genome = record[0]["GBSeq_sequence"]
                    if "complement" in coding_region["GBFeature_location"]:
                        genome = genome.complement()
                    cds = genome[cds_start : cds_end + 1]
                    sequences[type].append(cds)
                else:
                    protein = coding_region["GBFeature_quals"][-1]["GBQualifier_value"]
                    sequences[type].append(protein)

    @staticmethod
    def extract_genome_data_from_entrez_result(
        entrez_result: t.List[t.Dict],
    ) -> t.Tuple[t.Dict[str, str], t.Dict[str, str]]:
        virus_taxon_name_to_acc = dict()
        virus_taxon_name_to_seq = dict()
        for record in entrez_result:
            if (
                record["GBSeq_definition"]
                and record["GBSeq_organism"] not in virus_taxon_name_to_acc
                and record["GBSeq_organism"] not in virus_taxon_name_to_seq
            ):
                virus_taxon_name_to_acc[str(record["GBSeq_organism"]).lower()] = record[
                    "GBSeq_locus"
                ]
                virus_taxon_name_to_seq[str(record["GBSeq_organism"]).lower()] = str(
                    record["GBSeq_sequence"]
                )
        return virus_taxon_name_to_acc, virus_taxon_name_to_seq

    @staticmethod
    def get_gi_sequences(
        gi_accessions: t.List[str], batch_size: int = 500
    ) -> t.Dict[str, str]:
        """
        :param gi_accessions: list of gi accessions
        :return: dictionary mapping accessions to sequences
        """
        gi_accession_queries = [
            ",".join(gi_accessions[i : i + batch_size])
            for i in range(0, len(gi_accessions), batch_size)
        ]
        records = []
        for gi_accession_query in gi_accession_queries:
            records += list(
                Entrez.parse(
                    Entrez.efetch(db="nucleotide", id=gi_accession_query, retmode="xml")
                )
            )
        record_gi_acc_to_seq = dict()
        for record in records:
            for acc_data in record["GBSeq_other-seqids"]:
                if "gi" in acc_data:
                    acc = acc_data.split("|")[-1]
                    seq = record["GBSeq_sequence"]
                    record_gi_acc_to_seq[acc] = seq
        return record_gi_acc_to_seq

    @staticmethod
    def get_coding_regions(
        accessions: t.List[str], batch_size: int = 500
    ) -> t.Dict[str, str]:
        """
        :param accessions: list if refseq or genbank accessions
        :param batch_size: batch size for making queries to Entrez
        :return: dictionary mapping accession to a list of coding regions
        """
        queries = [
            ",".join(accessions[i : i + batch_size])
            for i in range(0, len(accessions), batch_size)
        ]
        records = []
        for query in queries:
            records += list(
                Entrez.parse(
                    Entrez.efetch(db="nucleotide", id=query, retmode="xml"),
                    validate=False,
                )
            )
        acc_to_coding_regions = {
            record["GBSeq_locus"]: ";".join(
                [
                    feature["GBFeature_location"]
                    for feature in record["GBSeq_feature-table"]
                    if feature["GBFeature_key"] == "CDS"
                ]
            )
            for record in records
        }
        return acc_to_coding_regions

    @staticmethod
    def extract_coding_sequence(
        genome_sequence: str, coding_regions: str
    ) -> t.List[str]:
        """
        :param genome_sequence: genomic sequence
        :param coding_regions: list of coding sequence regions in the form of join(a..c,c..d,...)
        :return: the coding sequence
        """
        coding_region_regex = re.compile("(\d*)\.\.(\d*)")
        coding_sequences = []
        for cds in coding_regions.split(";"):
            coding_sequence = ""
            for match in coding_region_regex.finditer(cds):
                start = int(match.group(1))
                end = int(match.group(2))
                assert (end - start) % 3 == 0
                coding_sequence += genome_sequence[start : end + 1]
            coding_sequences.append(coding_sequence)
        return coding_sequences


class GenomeBiasCollectingService:
    @staticmethod
    def get_dinucleotides_by_range(coding_sequence: str, seq_range: range):
        """
        :param coding_sequence: coding sequence
        :param seq_range: range for sequence window
        :return: a sequence of bridge / non-bridge dinucleotides depending on requested range
        """
        dinuc_sequence = "".join([coding_sequence[i : i + 2] for i in seq_range])
        return dinuc_sequence

    @staticmethod
    def compute_dinucleotide_bias(
        coding_sequences: t.List[str],
        computation_type: DinucleotidePositionType = DinucleotidePositionType.BRIDGE,
    ):
        """
        :param coding_sequences: list of coding sequences
        :param computation_type: can be either regular, or limited to bridge or non-bridge positions
        :return: dinucleotide bias dictionary
        dinculeotide bias computed according to https://science.sciencemag.org/content/sci/suppl/2018/10/31/362.6414.577.DC1/aap9072_Babayan_SM.pdf
        computation_type options:
            BRIDGE - consider only dinucleotide positions corresponding to bridges between codons (one is the last pos of a codon and the next is the first of another)
            NONBRIDGE - consider only dinucleotide positions do not correspond to bridges between codons
            REGULAR - consider all dinucleotide positions"""
        avg_dinucleotide_biases = dict()
        dinucleotide_biases_dicts = []
        for sequence in coding_sequences:
            dinuc_sequence = sequence
            if (
                computation_type == DinucleotidePositionType.BRIDGE
            ):  # limit the sequence to bridge positions only
                dinuc_sequence = GenomeBiasCollectingService.get_dinucleotides_by_range(
                    sequence, range(2, len(sequence) - 2, 3)
                )
            elif computation_type == DinucleotidePositionType.NONBRIDGE:
                dinuc_sequence = GenomeBiasCollectingService.get_dinucleotides_by_range(
                    sequence, range(0, len(sequence) - 2, 3)
                )
            nucleotide_count = {
                "A": dinuc_sequence.count("A"),
                "C": dinuc_sequence.count("C"),
                "G": dinuc_sequence.count("G"),
                "T": dinuc_sequence.count("T"),
            }
            nucleotide_total_count = len(dinuc_sequence)
            assert nucleotide_total_count > 0
            dinucleotide_total_count = len(sequence) / 2
            assert dinucleotide_total_count > 0
            dinucleotide_biases = dict()
            for nuc_i in nucleotide_count.keys():
                for nuc_j in nucleotide_count.keys():
                    dinucleotide = nuc_i + "p" + nuc_j
                    dinucleotide_biases[
                        computation_type.name + "_" + dinucleotide + "_bias"
                    ] = (sequence.count(dinucleotide) / dinucleotide_total_count) / (
                        nucleotide_count[nuc_i]
                        / nucleotide_total_count
                        * nucleotide_count[nuc_j]
                        / nucleotide_total_count
                    )
            dinucleotide_biases_dicts.append(dinucleotide_biases)

        # average dinucleotide biases across genomic sequences
        for dinucleotide in dinucleotide_biases_dicts[0]:
            avg_dinucleotide_biases[dinucleotide] = np.mean(
                [
                    dinucleotide_biases_dict[dinucleotide]
                    for dinucleotide_biases_dict in dinucleotide_biases_dicts
                ]
            )

        return avg_dinucleotide_biases

    @staticmethod
    def compute_codon_bias(coding_sequences: t.List[str]) -> t.Dict[str, float]:
        """
        :param coding_sequences: list of coding sequences
        :return: the codon bias computation described in https://science.sciencemag.org/content/sci/suppl/2018/10/31/362.6414.577.DC1/aap9072_Babayan_SM.pdf
        """
        sequence = "".join(coding_sequences)
        codon_biases = dict()
        for codon in CODONS:
            if codon not in STOP_CODONS:
                aa = Bio.Data.CodonTable.standard_dna_table.forward_table[codon]
                other_codons = [
                    codon
                    for codon in CODONS
                    if codon not in STOP_CODONS
                    and Bio.Data.CodonTable.standard_dna_table.forward_table[codon]
                    == aa
                ]
                codon_biases[codon + "_bias"] = sequence.count(codon) / np.sum(
                    [sequence.count(c) for c in other_codons]
                )
        return codon_biases

    @staticmethod
    def compute_diaa_bias(coding_sequences: t.List[str]) -> t.Dict[str, float]:
        """
        :param coding_sequences: list of coding sequences
        :return: the diaa biases, similar to compute_dinucleotide_bias
        """
        sequence = "".join([str(Seq(seq).translate()) for seq in coding_sequences])
        diaa_biases = dict()
        total_diaa_count = len(sequence) / 2
        total_aa_count = len(sequence)
        for aa_i in AMINO_ACIDS:
            for aa_j in AMINO_ACIDS:
                diaa = aa_i + aa_j
                diaa_biases[diaa + "_bias"] = (
                    sequence.count(diaa) / total_diaa_count
                ) / (
                    sequence.count(aa_i)
                    / total_aa_count
                    * sequence.count(aa_j)
                    / total_aa_count
                )
                if diaa_biases[diaa + "_bias"] == 0:
                    diaa_biases[diaa + "_bias"] += 0.0001
        return diaa_biases

    @staticmethod
    def compute_codon_pair_bias(
        coding_sequences: t.List[str], diaa_bias: t.Dict[str, float]
    ) -> t.Dict[str, float]:
        """
        :param coding_sequences: list of coding sequences
        :param diaa_bias: dictionary mapping diaa to its bias
        :return: dictionary mapping each dicodon to its bias
        codon pair bias measured by the codon pair score (CPS) as shown in https://science.sciencemag.org/content/sci/suppl/2018/10/31/362.6414.577.DC1/aap9072_Babayan_SM.pdf
        the denominator is obtained by multiplying the count od each codon with the bias of the respective amino acid pair
        """
        sequence = "".join(coding_sequences)
        codon_count = dict()
        for codon in CODONS:
            codon_count[codon] = sequence.count(codon)
        codon_pair_scores = dict()
        for codon_i in CODONS:
            for codon_j in CODONS:
                if codon_i not in STOP_CODONS and codon_j not in STOP_CODONS:
                    codon_pair = codon_i + codon_j
                    codon_pair_count = sequence.count(codon_pair)
                    denominator = (
                        codon_count[codon_i]
                        * codon_count[codon_j]
                        * diaa_bias[
                            f"{str(Seq(codon_i).translate())}{str(Seq(codon_j).translate())}_bias"
                        ]
                    )
                    if denominator == 0:
                        diaa_bias = diaa_bias[
                            f"{str(Seq(codon_i).translate())}{str(Seq(codon_j).translate())}_bias"
                        ]
                        print(
                            f"codon_count[{codon_i}]={codon_count[codon_i]}, codon_count[{codon_j}]={codon_count[codon_j]}, diaa_bias={diaa_bias}"
                        )
                        pass
                    else:
                        codon_pair_scores[codon_pair + "_bias"] = np.log(
                            codon_pair_count / denominator
                        )
        return codon_pair_scores

    @staticmethod
    def collect_genomic_bias_features(
        genome_sequences: t.List[str], coding_sequences: t.List[str]
    ):
        """
        :param genome_sequences: list of genomic sequences
        :param coding_sequences: list of coding sequences
        :return: dictionary with genomic features to be added as a record to a dataframe
        """
        dinucleotide_biases = GenomeBiasCollectingService.compute_dinucleotide_bias(
            coding_sequences=genome_sequences,
            computation_type=DinucleotidePositionType.REGULAR,
        )
        id_genomic_traits = dict(dinucleotide_biases)
        id_genomic_traits.update(
            GenomeBiasCollectingService.compute_dinucleotide_bias(
                coding_sequences=genome_sequences,
                computation_type=DinucleotidePositionType.BRIDGE,
            )
        )
        id_genomic_traits.update(
            GenomeBiasCollectingService.compute_dinucleotide_bias(
                coding_sequences=genome_sequences,
                computation_type=DinucleotidePositionType.NONBRIDGE,
            )
        )
        id_genomic_traits.update(
            GenomeBiasCollectingService.compute_codon_bias(
                coding_sequences=coding_sequences
            )
        )
        id_genomic_traits.update(
            GenomeBiasCollectingService.compute_diaa_bias(
                coding_sequences=coding_sequences
            )
        )
        id_genomic_traits.update(
            GenomeBiasCollectingService.compute_codon_pair_bias(
                coding_sequences=coding_sequences, diaa_bias=id_genomic_traits
            )
        )
        return id_genomic_traits
