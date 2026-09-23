import time

import sentencepiece
import transformers
import spacy_transformers

from opennyai.summarizer.ExtractiveSummarizer import ExtractiveSummarizer


class SummarizerRunner:

    def __init__(self, use_gpu=False, verbose=True, summary_length=0.0):
        self.use_gpu = use_gpu
        self.verbose = verbose
        self.summary_length = summary_length
        self.summarizer = None

    def load(self):
        if self.summarizer is not None:
            return

        print()
        print("=" * 80)
        print("LOADING EXTRACTIVE SUMMARIZER")
        print("=" * 80)

        start_time = time.perf_counter()

        # Required in the legacy OpenNyAI environment.
        _ = sentencepiece
        _ = transformers
        _ = spacy_transformers

        self.summarizer = ExtractiveSummarizer(
            use_gpu=self.use_gpu,
            verbose=self.verbose,
            summary_length=self.summary_length,
        )

        elapsed = time.perf_counter() - start_time

        print()
        print(f"Summarizer loaded in {elapsed:.2f} seconds")

    def run(self, rrc_result):
        if self.summarizer is None:
            self.load()

        print()
        print("=" * 80)
        print("RUNNING EXTRACTIVE SUMMARIZER")
        print("=" * 80)

        start_time = time.perf_counter()

        result = self.summarizer(rrc_result)

        elapsed = time.perf_counter() - start_time

        print()
        print(f"Summarization completed in {elapsed:.2f} seconds")

        return result