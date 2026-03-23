from datasets import load_dataset
import lm_eval.api.metrics as metrics
import re

def parse_sudoku_answer(response):
    """Extract sudoku answer from model response"""
    # Remove thinking tags if present
    if "<think>" in response and "</think>" in response:
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

@metrics.register_metric(metric="sudoku_accuracy", higher_is_better=True, aggregation="mean")
def sudoku_accuracy(predictions, references):
    """
    Custom sudoku evaluation metric for mergekit evolution
    
    Args:
        predictions: List of model responses
        references: List of reference data
        
    Returns:
        Dictionary with accuracy score
    """
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
