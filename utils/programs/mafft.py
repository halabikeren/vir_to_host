import logging
import os

logger = logging.getLogger(__name__)


class Mafft:
    @staticmethod
    def exec_mafft(input_path: str, output_path: str, threads_num: int = 1) -> int:
        """
        :param input_path: unaligned sequence data path
        :param output_path: aligned sequence data path
        :param threads_num: number of threads to use with mafft
        :return: return code
        """
        cmd = f"mafft --retree 1 --maxiterate 0 --thread {threads_num} {input_path} > {output_path}"
        res = os.system(cmd)
        if not os.path.exists(output_path):
            raise RuntimeError(f"failed to execute mafft on {input_path}")
        if res != 0:
            with open(output_path, "r") as outfile:
                outcontent = outfile.read()
            logger.error(f"failed mafft execution on {input_path} sequences from due to error {outcontent}")
        return res
