import re
import datasets

class SudokuVerifier:
    """Verify Sudoku solutions"""
    def verify(self, data: dict, test_solution: str) -> bool:
        """
        Verify if the solution is a valid completed sudoku

        Args:
            data: Game data containing the original puzzle
            test_solution: Model's response

        Returns:
            bool: True if valid solution, False otherwise
        """
        # Extract answer from response
        answer_str = self.extract_answer(test_solution)

        # Parse the answer grid
        answer_grid = self._parse_grid(answer_str)
        if answer_grid is None:
            return False

        # Parse the original puzzle
        puzzle_str = data.get("metadata", {}).get("puzzle", "")
        puzzle_grid = self._parse_grid(puzzle_str)
        if puzzle_grid is None:
            return False

        # Check 1: Answer respects the original puzzle (given cells match)
        if not self._respects_puzzle(answer_grid, puzzle_grid):
            return False

        # Check 2: Answer is a valid completed sudoku
        if not self._is_valid_sudoku(answer_grid):
            return False

        return True

    def extract_answer(self, test_solution: str) -> str:
        """
        Extract the sudoku grid from model response

        Looks for ```...``` block containing the grid, or 9 lines of 9 digits
        """
        # First try: find 9 lines of 9 digits anywhere in the response
        # This handles most common output formats
        lines = test_solution.strip().split('\n')
        grid_lines = []
        for line in lines:
            # Remove backticks and other common formatting
            clean_line = line.replace('`', '').strip()
            # Extract digits from line
            digits = re.findall(r'[1-9]', clean_line)
            if len(digits) == 9:
                grid_lines.append(''.join(digits))

        if len(grid_lines) >= 9:
            return '\n'.join(grid_lines[:9])

        # Fallback: try to find code block
        pattern = r"```([\s\S]*?)```"
        matches = re.findall(pattern, test_solution)

        if matches:
            # Get the last code block
            grid_str = matches[-1].strip()
            # Remove language identifier if present
            if grid_str and '\n' in grid_str:
                first_line = grid_str.split('\n')[0]
                if not re.search(r'[1-9]', first_line):
                    grid_str = '\n'.join(grid_str.split('\n')[1:])
            return grid_str

        return test_solution.strip()

    def _parse_grid(self, grid_str: str) -> list:
        """
        Parse grid string into 9x9 list of integers

        Args:
            grid_str: String representation of grid

        Returns:
            9x9 list of integers, or None if invalid
        """
        if not grid_str:
            return None

        lines = grid_str.strip().split('\n')
        grid = []

        for line in lines:
            line = line.strip()
            if not line:
                continue

            # Extract digits (1-9) and dots/zeros
            row = []
            for char in line:
                if char.isdigit():
                    row.append(int(char))
                elif char == '.':
                    row.append(0)
                # Skip other characters (spaces, separators, etc.)

            if len(row) == 9:
                grid.append(row)

        if len(grid) != 9:
            return None

        return grid

    def _respects_puzzle(self, answer: list, puzzle: list) -> bool:
        """
        Check if answer respects the given cells in puzzle

        Args:
            answer: 9x9 answer grid
            puzzle: 9x9 puzzle grid (0 for empty cells)

        Returns:
            bool: True if all given cells match
        """
        for i in range(9):
            for j in range(9):
                if puzzle[i][j] != 0:
                    if answer[i][j] != puzzle[i][j]:
                        return False
        return True

    def _is_valid_sudoku(self, grid: list) -> bool:
        """
        Check if grid is a valid completed sudoku

        Args:
            grid: 9x9 grid

        Returns:
            bool: True if valid
        """
        # Check all cells are filled with 1-9
        for i in range(9):
            for j in range(9):
                if grid[i][j] < 1 or grid[i][j] > 9:
                    return False

        # Check rows
        for i in range(9):
            if not self._is_valid_group(grid[i]):
                return False

        # Check columns
        for j in range(9):
            col = [grid[i][j] for i in range(9)]
            if not self._is_valid_group(col):
                return False

        # Check 3x3 boxes
        for box_row in range(3):
            for box_col in range(3):
                box = []
                for i in range(3):
                    for j in range(3):
                        box.append(grid[box_row * 3 + i][box_col * 3 + j])
                if not self._is_valid_group(box):
                    return False

        return True

    def _is_valid_group(self, group: list) -> bool:
        """
        Check if a group (row/column/box) contains exactly 1-9

        Args:
            group: List of 9 integers

        Returns:
            bool: True if contains exactly 1-9
        """
        return sorted(group) == list(range(1, 10))



def process_results(doc, results, game_data):
    prediction = results[0] if results else ""

    sudoku = SudokuVerifier()

    return {"accuracy": sudoku.verify(game_data, prediction)}

def preprocess(dataset: datasets.Dataset) -> datasets.Dataset:
    def _process_doc(doc):
        extra = doc["extra"]
        game_data = extra["game_data"]
        metadata = game_data.get("metadata") or extra.get("metadata") or {}

        return {
            "prompt": doc["prompt"],
            "game_data": game_data,
        }

    return dataset.map(_process_doc)
