# Author: gadwant
"""Tests for explanations, recommendations, multi_compare, bounds, profiles, docgen."""
import pytest

from bigocheck import benchmark_function
from bigocheck.explanations import (
    explain_complexity,
    get_complexity_comparison,
    suggest_improvement,
)
from bigocheck.recommendations import (
    suggest_sizes,
    detect_warmup_needed,
    SizeRecommendation,
)
from bigocheck.multi_compare import (
    compare_algorithms,
    generate_markdown_comparison,
)
from bigocheck.bounds import (
    check_bounds,
)
from bigocheck.profiles import (
    get_profile,
    benchmark_with_profile,
    list_profiles,
)
from bigocheck.docgen import (
    generate_complexity_docstring,
)


class TestExplanations:
    def test_explain_complexity_known(self):
        """Test explaining known complexity."""
        explanation = explain_complexity("O(n)")
        assert "Linear" in explanation
        assert "O(n)" in explanation
    
    def test_explain_complexity_unknown(self):
        """Test explaining unknown complexity."""
        explanation = explain_complexity("O(n^4)")
        assert "Polynomial" in explanation
    
    def test_get_complexity_comparison(self):
        """Test comparison table generation."""
        table = get_complexity_comparison()
        assert "O(1)" in table
        assert "O(n)" in table
        assert "Constant" in table
    
    def test_suggest_improvement(self):
        """Test improvement suggestions."""
        suggestion = suggest_improvement("O(n^2)")
        assert suggestion is not None
        assert "hash" in suggestion.lower() or "sort" in suggestion.lower()


class TestRecommendations:
    def test_suggest_sizes(self):
        """Test size suggestion."""
        def simple(n):
            return sum(range(n))
        
        rec = suggest_sizes(simple)
        assert isinstance(rec, SizeRecommendation)
        assert len(rec.sizes) >= 4
        assert rec.confidence in ["high", "medium", "low"]
    
    def test_detect_warmup_needed(self):
        """Test warmup detection."""
        def simple(n):
            return sum(range(n))
        
        # Simple function shouldn't need warmup
        result = detect_warmup_needed(simple)
        assert isinstance(result, bool)


class TestMultiCompare:
    def test_compare_algorithms(self):
        """Test comparing multiple algorithms."""
        def algo1(n):
            return sum(range(n))
        
        def algo2(n):
            total = 0
            for i in range(n):
                total += i
            return total
        
        result = compare_algorithms(
            {"algo1": algo1, "algo2": algo2},
            sizes=[100, 500],
        )
        
        assert result.winner is not None
        assert len(result.results) == 2
        assert "Algorithm" in result.summary_table
    
    def test_generate_markdown(self):
        """Test markdown generation."""
        def algo1(n):
            return sum(range(n))
        
        result = compare_algorithms({"algo1": algo1}, sizes=[100, 500])
        md = generate_markdown_comparison(result)
        
        assert "|" in md
        assert "Winner" in md


class TestBounds:
    def test_check_bounds_pass(self):
        """Test bounds check passes."""
        def simple(n):
            return sum(range(n))
        
        analysis = benchmark_function(simple, sizes=[100, 500, 1000])
        result = check_bounds(analysis, upper="O(n^2)")
        
        assert result.in_bounds
    
    def test_check_bounds_fail(self):
        """Test bounds check fails when exceeded."""
        def quadratic(n):
            for i in range(n):
                for j in range(n):
                    pass
        
        analysis = benchmark_function(quadratic, sizes=[50, 100, 200])
        result = check_bounds(analysis, upper="O(log n)")
        
        assert not result.in_bounds


class TestProfiles:
    def test_get_profile(self):
        """Test getting profiles."""
        profile = get_profile("fast")
        assert profile.name == "fast"
        assert len(profile.sizes) >= 3
    
    def test_get_profile_invalid(self):
        """Test invalid profile name."""
        with pytest.raises(ValueError):
            get_profile("nonexistent")
    
    def test_benchmark_with_profile(self):
        """Test benchmarking with profile."""
        def simple(n):
            return sum(range(n))
        
        analysis = benchmark_with_profile(simple, "fast")
        assert analysis.best_label is not None
    
    def test_list_profiles(self):
        """Test listing profiles."""
        listing = list_profiles()
        assert "fast" in listing
        assert "thorough" in listing


class TestDocgen:
    def test_generate_complexity_docstring(self):
        """Test docstring generation."""
        def simple(n):
            return sum(range(n))
        
        analysis = benchmark_function(simple, sizes=[100, 500])
        docstring = generate_complexity_docstring(analysis)
        
        assert "Complexity" in docstring
        assert "Time:" in docstring
