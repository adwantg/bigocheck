# Author: gadwant
"""Tests for new stability and badge features."""

from bigocheck import (
    benchmark_function,
)
from bigocheck.stability import compute_stability, format_stability, StabilityResult
from bigocheck.badges import (
    generate_badge,
    generate_dual_badge,
    save_badge,
    generate_badge_url,
)


class TestStabilityDetection:
    def test_compute_stability_returns_result(self):
        """Test compute_stability returns StabilityResult."""
        def simple(n):
            return sum(range(n))
        
        analysis = benchmark_function(simple, sizes=[100, 500, 1000], trials=3)
        result = compute_stability(analysis)
        
        assert isinstance(result, StabilityResult)
        assert 0.0 <= result.stability_score <= 1.0
        assert result.stability_level in ["high", "medium", "low"]
    
    def test_compute_stability_is_stable_inverse(self):
        """Test is_stable and is_unstable are inverses."""
        def simple(n):
            return sum(range(n))
        
        analysis = benchmark_function(simple, sizes=[100, 500, 1000, 5000], trials=5)
        result = compute_stability(analysis)
        
        assert result.is_stable == (not result.is_unstable)
    
    def test_format_stability(self):
        """Test format_stability produces readable output."""
        def simple(n):
            return sum(range(n))
        
        analysis = benchmark_function(simple, sizes=[100, 500, 1000], trials=3)
        result = compute_stability(analysis)
        output = format_stability(result)
        
        assert "Stability:" in output
        assert "%" in output


class TestBadgeGeneration:
    def test_generate_badge_svg(self):
        """Test generate_badge returns valid SVG."""
        svg = generate_badge("O(n log n)")
        
        assert svg.startswith("<svg")
        assert "O(n log n)" in svg
        assert "complexity" in svg  # Default label
    
    def test_generate_badge_custom_label(self):
        """Test generate_badge with custom label."""
        svg = generate_badge("O(n)", label="time")
        
        assert "time" in svg
        assert "O(n)" in svg
    
    def test_generate_badge_colors(self):
        """Test different complexities get different colors."""
        svg_o1 = generate_badge("O(1)")
        svg_on2 = generate_badge("O(n^2)")
        
        # O(1) should be green, O(n^2) should be red-ish
        assert "#4c1" in svg_o1  # Green
        assert "#e05d44" in svg_on2  # Red-orange
    
    def test_generate_dual_badge(self):
        """Test dual badge with time and space."""
        svg = generate_dual_badge("O(n)", "O(log n)")
        
        assert "O(n)" in svg
        assert "O(log n)" in svg
    
    def test_generate_badge_url(self):
        """Test shields.io URL generation."""
        url = generate_badge_url("O(n log n)")
        
        assert "shields.io" in url
        assert "complexity" in url
        assert "O%28n%20log%20n%29" in url  # URL encoded


class TestBadgeSave:
    def test_save_badge(self, tmp_path):
        """Test saving badge to file."""
        svg = generate_badge("O(n)")
        path = tmp_path / "badge.svg"
        
        save_badge(svg, str(path))
        
        assert path.exists()
        content = path.read_text()
        assert "<svg" in content
