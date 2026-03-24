import re
import datasets

class TraceVerifier:
    """Verify Print solutions"""
    def __init__(self):
        pass

    def verify(self, extra: dict, response: str) -> int:
        ground_truth = extra.get("ground_truth", "")
        cleaned_prediction = self.clean_llm_prediction(response)

        score = 1.0 if self.compare_outputs(ground_truth, cleaned_prediction) else 0.0

        return score

    def clean_llm_prediction(self, prediction: str) -> str:
        """Remove reasoning/thinking and markdown code blocks from LLM response"""
        prediction = re.sub(r"<think>.*?</think>", "", prediction, flags=re.DOTALL)
        prediction = re.sub(r"<thinking>.*?</thinking>", "", prediction, flags=re.DOTALL)

        if "</think>" in prediction:
            prediction = prediction.split("</think>")[-1].strip()
        if "</thinking>" in prediction:
            prediction = prediction.split("</thinking>")[-1].strip()

        code_block_match = re.search(r"```(?:[a-zA-Z]*)\n?(.*?)\n?```", prediction, flags=re.DOTALL)
        if code_block_match:
            prediction = code_block_match.group(1)
        
        return prediction.strip()

    def compare_outputs(self, expected: str, actual: str) -> bool:
        """
        Compare two strings for equality, but with leniency:
        1. Normalize line endings
        2. Ignore all internal whitespace differences
        3. Case-insensitive comparison
        """
        def normalize(s):
            # Remove all whitespace and convert to lowercase
            return "".join(s.split()).lower()
        
        return normalize(expected) == normalize(actual)


def process_results(doc, results, extra):
    prediction = results[0] if results else ""

    verifier = TraceVerifier()

    return {"accuracy": verifier.verify(extra, prediction)}

def preprocess(dataset: datasets.Dataset) -> datasets.Dataset:
    def _process_doc(doc):
        extra = doc["extra"]

        return {
            "prompt": doc["prompt"],
            "extra": extra,
        }

    return dataset.map(_process_doc)
