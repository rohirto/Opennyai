import time

import sentencepiece
import transformers
import spacy_transformers

from opennyai import Pipeline
from opennyai.utils import Data


class RRCRunner:
    """
    Reusable wrapper around the OpenNyAI Rhetorical Role pipeline.

    The OpenNyAI RRC model is loaded once and reused for
    multiple judgment documents.
    """

    PREPROCESSING_MODEL = "en_core_web_sm"

    def __init__(
        self,
        use_gpu=False,
        verbose=True,
    ):
        self.use_gpu = use_gpu
        self.verbose = verbose

        self.pipeline = None

    def load(self):
        """
        Load the OpenNyAI Rhetorical Role pipeline once.
        """

        if self.pipeline is not None:
            return

        print()
        print("=" * 80)
        print("LOADING RHETORICAL ROLE PIPELINE")
        print("=" * 80)

        start_time = time.perf_counter()

        # These imports must happen before OpenNyAI initialization
        # in the current legacy Windows environment.
        _ = sentencepiece
        _ = transformers
        _ = spacy_transformers

        self.pipeline = Pipeline(
            components=["Rhetorical_Role"],
            use_gpu=self.use_gpu,
            verbose=self.verbose,
        )

        elapsed = time.perf_counter() - start_time

        print()
        print(
            f"Pipeline loaded in {elapsed:.2f} seconds"
        )

    def run(self, document):
        """
        Run Rhetorical Role Classification on a LegalDocument.

        Parameters
        ----------
        document:
            LegalDocument instance.

        Returns
        -------
        result:
            OpenNyAI RRC result.
        """

        if self.pipeline is None:
            self.load()

        print()
        print("=" * 80)
        print(
            f"OPENNYAI DATA PREPROCESSING: "
            f"{document.document_id}"
        )
        print("=" * 80)

        print(
            f"Preprocessing model : "
            f"{self.PREPROCESSING_MODEL}"
        )

        print(
            f"GPU                 : "
            f"{self.use_gpu}"
        )

        # ---------------------------------------------------------------
        # OpenNyAI Data preprocessing
        # ---------------------------------------------------------------

        start_time = time.perf_counter()

        data = Data(
            document.cleaned_text,
            preprocessing_nlp_model=self.PREPROCESSING_MODEL,
            use_gpu=self.use_gpu,
            verbose=self.verbose,
        )

        data_time = time.perf_counter() - start_time

        print()
        print(
            f"Data object created in {data_time:.2f} seconds"
        )

        # ---------------------------------------------------------------
        # Rhetorical Role inference
        # ---------------------------------------------------------------

        print()
        print("=" * 80)
        print(
            f"RUNNING RHETORICAL ROLE INFERENCE: "
            f"{document.document_id}"
        )
        print("=" * 80)

        inference_start = time.perf_counter()

        result = self.pipeline(data)

        inference_time = (
            time.perf_counter() - inference_start
        )

        print()
        print(
            f"Inference completed in "
            f"{inference_time:.2f} seconds"
        )

        return result