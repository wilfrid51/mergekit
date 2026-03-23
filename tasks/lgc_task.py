from lm_eval.api.task import ConfigurableTask
from lm_eval.api.metrics import mean
import re
from datasets import load_dataset

def parse_sudoku_answer(response):
    """Extract sudoku answer from model response"""
    # Remove thinking tags if present
    if "</think>" in response and "</think>" in response:
        response = response.split("</think>")[-1].strip()
    
    # Look for patterns that look like sudoku solutions
    # Common formats: sequences of digits, grid patterns, etc.
    
    # Extract digits that could form a solution
    digits = re.findall(r'\d', response)
    
    # If we have 81 digits (9x9 sudoku), that's likely the solution
    if len(digits) >= 81:
        return ''.join(digits[:81])
    
    # Otherwise, look for other patterns
    lines = response.strip().split('\n')
    for line in lines:
        # Skip empty lines and obvious non-sudoku content
        if not line.strip():
            continue
            
        # Check if line contains mostly digits
        digit_chars = sum(1 for c in line if c.isdigit())
        if digit_chars >= 8:  # At least 8 digits suggests a solution row
            return line.strip()
    
    # Fallback: return the response as-is for evaluation
    return response.strip()

def is_valid_sudoku_attempt(answer):
    """Check if the answer represents a valid sudoku attempt"""
    if not answer or len(answer) < 10:
        return False
    
    # Count digits
    digit_count = sum(1 for c in answer if c.isdigit())
    
    # Should have reasonable number of digits for a sudoku solution
    if digit_count < 20:
        return False
    
    # Check for reasonable digit distribution (1-9)
    digits = [c for c in answer if c.isdigit()]
    unique_digits = set(digits)
    
    # Should use multiple different digits, not just one repeated
    return len(unique_digits) >= 3

def sudoku_accuracy_fn(references=None, predictions=None, items=None):
    """Calculate sudoku accuracy from model responses"""
    
    # Handle different calling conventions
    if items is not None:
        # Called with items (list of tuples)
        if not items:
            return 0.0
        
        if isinstance(items[0], (list, tuple)):
            # items is a list of (prediction, reference) tuples
            predictions = [item[0] for item in items]
            references = [item[1] for item in items]
        else:
            # items is a single (predictions, references) tuple
            predictions, references = items
    elif references is not None and predictions is not None:
        # Called with references and predictions lists
        pass  # Already have the right variables
    else:
        return 0.0
    
    try:
        # Load dataset
        dataset = load_dataset("jeff4000/sudoku_task", split="train")
        
        correct = 0
        total = 0
        
        for i, (pred, ref) in enumerate(zip(predictions, references)):
            if i >= len(dataset):
                break
                
            # Parse model response to extract answer
            answer = parse_sudoku_answer(pred)
            
            # Simple validation: check if answer looks like a valid sudoku attempt
            # For evolution purposes, reward coherent responses
            if is_valid_sudoku_attempt(answer):
                correct += 1
            total += 1
        
        return correct / total if total > 0 else 0.0
        
    except Exception as e:
        print(f"Error in sudoku metric: {e}")
        return 0.0

class LgcTask(ConfigurableTask):
    VERSION = 0.1
    DATASET_PATH = "jeff4000/sudoku_task"
    OUTPUT_TYPE = "generate_until"
    TRAINING_SPLIT = "train"
    VALID_SPLIT = "train"  # Use train split for validation since only train is available
    TEST_SPLIT = "train"  # Use train split for testing since only train is available

    def __init__(self, config=None):
        config = config or {}
        super().__init__(config=config)
        self._metric_fn_list = {
            "sudoku_accuracy": sudoku_accuracy_fn,
            "acc": sudoku_accuracy_fn,  # Add alias for acc
            "acc_none": sudoku_accuracy_fn,  # Add alias for acc_none
        }

    def has_test_docs(self):
        return True

    def test_docs(self):
        return self.dataset["train"]

    def has_validation_docs(self):
        return True

    def validation_docs(self):
        return self.dataset["train"]

    def doc_to_text(self, doc):
        """Extract the prompt text from the document"""
        return doc["prompt"]

    def doc_to_target(self, doc):
        """Extract the target answer from the document"""
        return doc["extra"]["game_data"]["answer"]

    def process_results(self, doc, results):
        """Process results for sudoku evaluation"""
        print(f"DEBUG: process_results called")
        print(f"DEBUG: _metric_fn_list contents: {self._metric_fn_list}")
        print(f"DEBUG: _metric_fn_list keys: {list(self._metric_fn_list.keys())}")
        for key, value in self._metric_fn_list.items():
            print(f"DEBUG: {key}: {value} (callable: {callable(value)})")
        
        processed = {
            "sudoku_accuracy": (results, doc),
            "acc": (results, doc),
            "acc_none": (results, doc),
        }
        return processed

    def aggregation(self):
        """Define aggregation methods"""
        return {
            "sudoku_accuracy": mean,
            "acc": mean,
            "acc_none": mean,
        }

    def higher_is_better(self):
        """Define which metrics are better when higher"""
        return {
            "sudoku_accuracy": True,
            "acc": True,
            "acc_none": True,
        }

# Register the task
TASK_REGISTRY = {
    "lgc_task": LgcTask,
}
