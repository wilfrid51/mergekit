import re
import datasets

class TraceVerifier:
    """Verify Print solutions"""
    def __init__(self):
        pass

    def verify(self, extra: dict, response: str) -> int:
        ground_truth = extra.get("ground_truth", "")
        print(f"{'='*50} GROUND_TRUTH {'='*50}\n{ground_truth}")
        print(f"{'='*50}   RESPONSE {'='*50}\n{response}")
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


def check_results(doc, results):
    prediction = results[0] if results else ""
    extra = doc["extra"]

    verifier = TraceVerifier()
    score = verifier.verify(extra, prediction)

    print(f"{'='*100}\nCryptarithm Score: {score} - {len(prediction) * 0.00001} = {score - len(prediction) * 0.00001}")
    
    return {"accuracy": score - len(prediction) * 0.00001}

def preprocess(dataset: datasets.Dataset) -> datasets.Dataset:
    def _process_doc(doc):
        extra = doc["extra"]

        return {
            "prompt": doc["prompt"],
            "extra": extra,
        }

    return dataset.map(_process_doc)
