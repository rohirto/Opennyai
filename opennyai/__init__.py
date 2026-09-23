import importlib


def __getattr__(name):
    if name == "ner":
        return importlib.import_module(".ner", __name__)

    if name == "utils":
        return importlib.import_module(".utils", __name__)

    if name == "Pipeline":
        from opennyai.pipeline import Pipeline
        return Pipeline

    if name == "RhetoricalRolePredictor":
        from opennyai.rhetorical_roles.rhetorical_roles import RhetoricalRolePredictor
        return RhetoricalRolePredictor

    if name == "ExtractiveSummarizer":
        from opennyai.summarizer.ExtractiveSummarizer import ExtractiveSummarizer
        return ExtractiveSummarizer

    raise AttributeError(
        f"module {__name__!r} has no attribute {name!r}"
    )