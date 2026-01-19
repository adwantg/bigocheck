# Author: gadwant
"""Tests for cases module (best/worst/average case analysis)."""

from bigocheck import analyze_cases, CaseResult, CasesAnalysis, format_cases_result


class TestAnalyzeCases:
    def test_analyze_cases_returns_result(self):
        """Test that analyze_cases returns a CasesAnalysis."""
        def my_sort(arr):
            return sorted(arr)
        
        result = analyze_cases(my_sort, sizes=[100, 500], trials=1)
        
        assert isinstance(result, CasesAnalysis)
        assert "best" in result.results
        assert "worst" in result.results
        assert "average" in result.results
    
    def test_case_results_have_analysis(self):
        """Test that each case has an Analysis object."""
        def my_sort(arr):
            return sorted(arr)
        
        result = analyze_cases(my_sort, sizes=[100, 500], trials=1)
        
        for case_name, case_result in result.results.items():
            assert isinstance(case_result, CaseResult)
            assert case_result.analysis is not None
            assert case_result.time_complexity is not None
    
    def test_best_worst_identified(self):
        """Test that best and worst cases are identified."""
        def my_sort(arr):
            return sorted(arr)
        
        result = analyze_cases(my_sort, sizes=[100, 500], trials=1)
        
        assert result.best_case is not None
        assert result.worst_case is not None
        assert result.best_case.avg_time <= result.worst_case.avg_time
    
    def test_custom_cases(self):
        """Test with custom case generators."""
        def my_func(data):
            return sum(data)
        
        custom_cases = {
            "small": lambda n: list(range(n // 10)),
            "large": lambda n: list(range(n)),
        }
        
        result = analyze_cases(my_func, sizes=[100, 500], cases=custom_cases, trials=1)
        
        assert "small" in result.results
        assert "large" in result.results
        assert len(result.results) == 2
    
    def test_summary_generated(self):
        """Test that summary is generated."""
        def my_sort(arr):
            return sorted(arr)
        
        result = analyze_cases(my_sort, sizes=[100, 500], trials=1)
        
        assert result.summary is not None
        assert "Case Analysis Summary" in result.summary


class TestFormatCasesResult:
    def test_format_cases_result(self):
        """Test formatting of cases result."""
        def my_sort(arr):
            return sorted(arr)
        
        result = analyze_cases(my_sort, sizes=[100, 500], trials=1)
        formatted = format_cases_result(result)
        
        assert "CASE ANALYSIS" in formatted
        assert "BEST" in formatted
        assert "WORST" in formatted
        assert "AVERAGE" in formatted
