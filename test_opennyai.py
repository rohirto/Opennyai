from opennyai import Pipeline
from opennyai.utils import Data

text = """
The petitioner filed the present appeal against the judgment of the High Court.
The learned counsel for the petitioner argued that the impugned order was contrary
to the provisions of the Indian Penal Code. The respondent opposed the appeal.
We have considered the submissions made by learned counsel for the parties.
In the result, the appeal is dismissed.
"""

data = Data([text])

pipeline = Pipeline(
    components=['NER'],
    use_gpu=False,
    verbose=True,
    ner_model_name='en_legal_ner_trf',
    ner_do_sentence_level=True,
    ner_do_postprocess=True,
    ner_statute_shortforms_path=''
)

results = pipeline(data)

print("\n=== RESULTS ===")
print(results)

print("\n=== ENTITIES ===")
ner_doc = pipeline._ner_model_output[0]

for ent in ner_doc.ents:
    print(repr(ent.text), "=>", ent.label_)

print("\n=== PRECEDENT CLUSTERS ===")
print(ner_doc.user_data.get("precedent_clusters"))

print("\n=== STATUTE CLUSTERS ===")
print(ner_doc.user_data.get("statute_clusters"))
