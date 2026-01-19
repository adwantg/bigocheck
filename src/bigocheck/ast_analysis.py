# Author: gadwant
"""
Static complexity analysis using Python's AST (Abstract Syntax Tree).
Provides a "theoretical" Complexity Estimate to compare with empirical results.
"""
import ast
import inspect
import textwrap
from typing import Callable, Dict

class ComplexityPredictor(ast.NodeVisitor):
    def __init__(self):
        self.complexities = []
        self.current_depth = 0
        self.max_depth = 0
        self.has_recursion = False
        self.func_name = ""

    def visit_FunctionDef(self, node):
        if not self.func_name:
            self.func_name = node.name
        self.generic_visit(node)

    def visit_For(self, node):
        self.current_depth += 1
        self.max_depth = max(self.max_depth, self.current_depth)
        self.generic_visit(node)
        self.current_depth -= 1

    def visit_While(self, node):
        self.current_depth += 1
        self.max_depth = max(self.max_depth, self.current_depth)
        self.generic_visit(node)
        self.current_depth -= 1

    def visit_ListComp(self, node):
        # List comprehensions are effectively loops
        self.current_depth += 1
        self.max_depth = max(self.max_depth, self.current_depth)
        self.generic_visit(node)
        self.current_depth -= 1

    def visit_Call(self, node):
        # Check for recursion
        if isinstance(node.func, ast.Name) and node.func.id == self.func_name:
            self.has_recursion = True
        self.generic_visit(node)

def predict_complexity(func: Callable) -> Dict[str, str]:
    """
    Statically analyze a function to predict its Time Complexity.
    
    Returns:
        Dict with keys: 'prediction', 'reason', 'confidence'
    """
    try:
        source = inspect.getsource(func)
        source = textwrap.dedent(source)
        tree = ast.parse(source)
    except (OSError, TypeError, SyntaxError):
        return {
            "prediction": "Unknown",
            "reason": "Could not parse source code (dynamic or compiled function)",
            "confidence": "none"
        }

    visitor = ComplexityPredictor()
    visitor.visit(tree)

    # Heuristic Logic
    if visitor.has_recursion:
        # Recursion is hard to guess statically without flow analysis
        # But usually at least O(log n) or O(n) or O(2^n)
        return {
            "prediction": "O(log n)+",
            "reason": "Recursive calls detected (could be O(n log n) or O(2^n))",
            "confidence": "low"
        }
    
    if visitor.max_depth == 0:
        return {
            "prediction": "O(1)",
            "reason": "No loops or list comprehensions detected",
            "confidence": "high"
        }
    elif visitor.max_depth == 1:
        return {
            "prediction": "O(n)",
            "reason": "Single loop detected",
            "confidence": "medium"
        }
    elif visitor.max_depth == 2:
        return {
            "prediction": "O(n^2)",
            "reason": "Nested loops (depth 2) detected",
            "confidence": "medium"
        }
    else:
        return {
            "prediction": f"O(n^{visitor.max_depth})",
            "reason": f"Deeply nested loops (depth {visitor.max_depth}) detected",
            "confidence": "medium"
        }

def verify_hybrid(func: Callable, empirical_result: str) -> str:
    """Compare static prediction with empirical result."""
    prediction = predict_complexity(func)
    pred_val = prediction["prediction"]
    
    if pred_val == "Unknown":
        return f"Static analysis failed: {prediction['reason']}"
    
    # Normalize for comparison
    clean_emp = empirical_result.replace(" ", "")
    clean_pred = pred_val.replace(" ", "")
    
    if clean_pred in clean_emp or clean_emp in clean_pred:
        return f"✅ Match! Static ({pred_val}) aligns with Empirical ({empirical_result})"
    else:
        return f"⚠️ Mismatch. Static predicted {pred_val}, but runtime was {empirical_result}. ({prediction['reason']})"
