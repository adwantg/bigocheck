import os
import shutil
import tempfile
from bigocheck.ast_analysis import predict_complexity, verify_hybrid
from bigocheck.dashboard import generate_dashboard
from bigocheck.cloud import generate_github_action
from bigocheck.core import Analysis

# Tiny helpers for AST testing
def func_constant(n):
    return n * 2

def func_linear(n):
    for i in range(n):
        pass

def func_quadratic(n):
    for i in range(n):
        for j in range(n):
            pass

def func_recursive(n):
    if n <= 1: return 1
    return func_recursive(n-1)

def test_ast_constant():
    pred = predict_complexity(func_constant)
    assert pred["prediction"] == "O(1)"
    assert pred["confidence"] == "high"

def test_ast_linear():
    pred = predict_complexity(func_linear)
    assert pred["prediction"] == "O(n)"
    assert pred["confidence"] == "medium"

def test_ast_quadratic():
    pred = predict_complexity(func_quadratic)
    assert pred["prediction"] == "O(n^2)"

def test_ast_recursive():
    pred = predict_complexity(func_recursive)
    assert "log n" in pred["prediction"] or "2^n" in pred["prediction"] or "Recursive" in pred["reason"]

def test_ast_unparsable():
    # Built-ins often can't be parsed by inspect.getsource
    pred = predict_complexity(sum)
    assert pred["prediction"] == "Unknown"
    assert pred["confidence"] == "none"

def test_hybrid_verification():
    # Match
    res = verify_hybrid(func_linear, "O(n)")
    assert "✅" in res
    
    # Mismatch
    res = verify_hybrid(func_linear, "O(1)")
    assert "⚠️" in res

def test_dashboard_generation():
    tmp_dir = tempfile.mkdtemp()
    try:
        # Create dummy analysis
        analysis = Analysis(
            measurements=[], 
            fits=[], 
            best_label="O(n)", 
            name="test_func"
        )
        
        generate_dashboard([analysis], output_dir=tmp_dir)
        
        assert os.path.exists(os.path.join(tmp_dir, "index.html"))
        
        with open(os.path.join(tmp_dir, "index.html")) as f:
            content = f.read()
            assert "test_func" in content
            assert "O(n)" in content
    finally:
        shutil.rmtree(tmp_dir)

def test_cloud_generator():
    tmp_dir = tempfile.mkdtemp()
    try:
        # 1. Generate
        generate_github_action(output_dir=tmp_dir)
        yaml_path = os.path.join(tmp_dir, "bigocheck_benchmark.yml")
        assert os.path.exists(yaml_path)
        
        with open(yaml_path) as f:
            content = f.read()
            assert "uses: actions/setup-python" in content
            
        # 2. Idempotency (should not overwrite if exists - simulated by checking print/return)
        # Ideally we'd test logic, but for now just ensure it doesn't crash
        generate_github_action(output_dir=tmp_dir)
        
    finally:
        shutil.rmtree(tmp_dir)
