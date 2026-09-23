from pathlib import Path
import time

import sentencepiece
import transformers
import spacy_transformers
import spacy

from opennyai import Pipeline


class NERRunner:
    """
    Wrapper around the legacy OpenNyAI NER pipeline.

    The model is loaded once and can then process multiple judgments.
    """

    def __init__(self, use_gpu=False, verbose=True):
        self.use_gpu = use_gpu
        self.verbose = verbose

        self.pipeline = None
        self.nlp = None

    def load(self):
        """Load the OpenNyAI NER model."""

        if self.pipeline is not None:
            return

        print("\n" + "=" * 80)
        print("LOADING OPENNYAI NER")
        print("=" * 80)

        start = time.perf_counter()

        # Required in the legacy Windows environment.
        # These imports must happen before OpenNyAI model initialization.
        _ = sentencepiece
        _ = transformers
        _ = spacy_transformers

        self.nlp = spacy.blank("en")
        self.nlp.add_pipe("sentencizer")

        self.pipeline = Pipeline(
            components=["NER"],
            use_gpu=self.use_gpu,
            verbose=self.verbose,
        )

        elapsed = time.perf_counter() - start

        print(
            f"NER model loaded in {elapsed:.2f} seconds"
        )

    def run(self, document):
        """
        Run NER on a LegalDocument.

        Parameters
        ----------
        document:
            LegalDocument instance.

        Returns
        -------
        list
            OpenNyAI NER result.
        """

        if self.pipeline is None:
            self.load()

        print(
            f"\nRunning NER: {document.document_id}"
        )

        start = time.perf_counter()

        # OpenNyAI's legacy NER implementation expects
        # both judgement_doc and preamble_doc.
        #
        # We deliberately DO NOT perform semantic
        # preamble/judgment splitting here.
        judgement_doc = self.nlp(
            document.cleaned_text
        )

        empty_preamble_doc = self.nlp("")

        data = [
            {
                "judgement_doc": judgement_doc,
                "preamble_doc": empty_preamble_doc,
                "file_id": document.document_id,
                "original_text": document.cleaned_text,
            }
        ]

        result = self.pipeline(data)

        elapsed = time.perf_counter() - start

        entities = []

        for output_document in result:
            for annotation in output_document.get(
                "annotations",
                []
            ):
                entities.extend(
                    annotation.get(
                        "entities",
                        []
                    )
                )

        print(
            f"NER completed in {elapsed:.2f} seconds"
        )

        print(
            f"Entities detected: {len(entities):,}"
        )

        return result