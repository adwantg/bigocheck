# Author: gadwant
"""
Command-line interface for bigocheck.
"""
from __future__ import annotations

import argparse
import json
import sys
from typing import List

from .core import Analysis, benchmark_function, resolve_callable


def _format_scale(scale: float) -> str:
    """Format time scale in human-readable units."""
    if scale < 0:
        return f"{scale:.4g}"
    
    if scale >= 1:
        return f"{scale:.2f} s/elem"
    elif scale >= 1e-3:
        return f"{scale * 1e3:.2f} ms/elem"
    elif scale >= 1e-6:
        return f"{scale * 1e6:.2f} µs/elem"
    elif scale >= 1e-9:
        return f"{scale * 1e9:.2f} ns/elem"
    else:
        return f"{scale:.2e}"


def _format_scale_bytes(scale: float) -> str:
    """Format memory scale in human-readable units."""
    if scale < 0:
        return f"{scale:.4g}"
    
    if scale >= 1e9:
        return f"{scale / 1e9:.2f} GB/elem"
    elif scale >= 1e6:
        return f"{scale / 1e6:.2f} MB/elem"
    elif scale >= 1e3:
        return f"{scale / 1e3:.2f} KB/elem"
    else:
        return f"{scale:.2f} B/elem"


def _parse_args(argv: List[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Empirical complexity regression checker.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  bigocheck run --target mymodule:myfunc --sizes 100 500 1000
  bigocheck run --target mymodule:myfunc --sizes 100 500 1000 --trials 5 --warmup 2
  bigocheck run --target mymodule:myfunc --sizes 100 500 1000 --json
  bigocheck run --target mymodule:myfunc --sizes 100 500 1000 --verbose --memory
  bigocheck regression --target mymodule:myfunc --baseline baseline.json
  bigocheck repl
""",
    )
    sub = parser.add_subparsers(dest="command", required=True)

    # Run command
    run_parser = sub.add_parser("run", help="Run a benchmark against a callable.")
    run_parser.add_argument(
        "--target",
        required=True,
        help="Import path in the form module:func",
    )
    run_parser.add_argument(
        "--sizes",
        required=True,
        nargs="+",
        type=int,
        help="Input sizes to measure",
    )
    run_parser.add_argument(
        "--trials",
        type=int,
        default=3,
        help="Number of runs per size (averaged). Default: 3",
    )
    run_parser.add_argument(
        "--warmup",
        type=int,
        default=0,
        help="Number of warmup runs before timing. Default: 0",
    )
    run_parser.add_argument(
        "--json",
        action="store_true",
        help="Emit JSON output instead of human-readable format",
    )
    run_parser.add_argument(
        "--verbose",
        "-v",
        action="store_true",
        help="Show progress during benchmarking",
    )
    run_parser.add_argument(
        "--memory",
        action="store_true",
        help="Track peak memory usage and compute space complexity",
    )
    run_parser.add_argument(
        "--plot",
        action="store_true",
        help="Generate a plot of the results (requires matplotlib)",
    )
    run_parser.add_argument(
        "--plot-save",
        type=str,
        default=None,
        metavar="PATH",
        help="Save plot to file instead of displaying",
    )
    run_parser.add_argument(
        "--html",
        type=str,
        default=None,
        metavar="PATH",
        help="Generate HTML report and save to file",
    )
    run_parser.add_argument(
        "--save-baseline",
        type=str,
        default=None,
        metavar="PATH",
        help="Save results as baseline for regression detection",
    )

    # Regression command
    reg_parser = sub.add_parser("regression", help="Check for performance regressions.")
    reg_parser.add_argument(
        "--target",
        required=True,
        help="Import path in the form module:func",
    )
    reg_parser.add_argument(
        "--baseline",
        required=True,
        help="Path to baseline JSON file",
    )
    reg_parser.add_argument(
        "--sizes",
        nargs="+",
        type=int,
        default=None,
        help="Input sizes (uses baseline sizes if not specified)",
    )
    reg_parser.add_argument(
        "--threshold",
        type=float,
        default=0.2,
        help="Slowdown threshold (0.2 = 20%%). Default: 0.2",
    )

    # REPL command
    sub.add_parser("repl", help="Start interactive REPL mode.")

    # Explain command
    explain_parser = sub.add_parser("explain", help="Explain a complexity class.")
    explain_parser.add_argument(
        "complexity",
        help="Complexity label (e.g., 'O(n log n)')",
    )

    # Recommend command
    rec_parser = sub.add_parser("recommend", help="Recommend input sizes for a function.")
    rec_parser.add_argument(
        "--target",
        required=True,
        help="Import path in the form module:func",
    )
    rec_parser.add_argument(
        "--time-budget",
        type=float,
        default=5.0,
        help="Maximum time budget in seconds (default: 5.0)",
    )

    # Compare command
    comp_parser = sub.add_parser("compare", help="Compare multiple algorithms.")
    comp_parser.add_argument(
        "--targets",
        required=True,
        nargs="+",
        help="List of functions to compare (module:func)",
    )
    comp_parser.add_argument(
        "--sizes",
        required=True,
        nargs="+",
        type=int,
        help="Input sizes to benchmark",
    )
    comp_parser.add_argument(
        "--trials",
        type=int,
        default=3,
        help="Number of trials per size",
    )
    comp_parser.add_argument(
        "--json",
        action="store_true",
        help="Output JSON summary",
    )

    # Dashboard command
    dash_parser = sub.add_parser("dashboard", help="Generate static web dashboard.")
    dash_parser.add_argument(
        "--targets",
        required=True,
        nargs="+",
        help="List of functions to benchmark (module:func)",
    )
    dash_parser.add_argument(
        "--output",
        default="dashboard",
        help="Output directory (default: dashboard)",
    )
    dash_parser.add_argument(
        "--sizes",
        nargs="+",
        type=int,
        default=[100, 500, 1000],
        help="Input sizes to benchmark",
    )

    # Cloud command
    sub.add_parser("cloud", help="Generate GitHub Actions workflow.")


def _analysis_to_json(analysis: Analysis) -> str:
    data = {
        "time_complexity": analysis.best_label,
        "space_complexity": analysis.space_label,
        "measurements": [
            {
                "size": m.size,
                "seconds": m.seconds,
                "std_dev": m.std_dev,
                **({} if m.memory_bytes is None else {"memory_bytes": m.memory_bytes}),
            }
            for m in analysis.measurements
        ],
        "time_fits": [
            {"label": f.label, "scale": f.scale, "error": f.error}
            for f in analysis.fits
        ],
    }
    
    # Add space fits if available
    if analysis.space_fits:
        data["space_fits"] = [
            {"label": f.label, "scale": f.scale, "error": f.error}
            for f in analysis.space_fits
        ]
    
    return json.dumps(data, indent=2)


def _print_human(analysis: Analysis, show_memory: bool = False) -> None:
    # Header with complexity results
    print(f"Time Complexity:  {analysis.best_label}")
    if show_memory and analysis.space_label:
        print(f"Space Complexity: {analysis.space_label}")
    
    print("\nMeasurements:")
    
    if show_memory and any(m.memory_bytes for m in analysis.measurements):
        for m in analysis.measurements:
            mem_str = f"  mem={m.memory_bytes:,}B" if m.memory_bytes else ""
            print(f"  n={m.size:<8} time={m.seconds:.6f}s ±{m.std_dev:.6f}s{mem_str}")
    else:
        for m in analysis.measurements:
            print(f"  n={m.size:<8} time={m.seconds:.6f}s ±{m.std_dev:.6f}s")
    
    print("\nTime Fits (lower error is better):")
    for f in analysis.fits[:5]:  # Show top 5
        marker = " ★" if f.label == analysis.best_label else ""
        print(f"  {f.label:<12} error={f.error:.4f} scale={_format_scale(f.scale)}{marker}")
    
    # Show space fits if available
    if show_memory and analysis.space_fits:
        print("\nSpace Fits (lower error is better):")
        for f in analysis.space_fits[:5]:  # Show top 5
            marker = " ★" if f.label == analysis.space_label else ""
            print(f"  {f.label:<12} error={f.error:.4f} scale={_format_scale_bytes(f.scale)}{marker}")


def main(argv: List[str] | None = None) -> None:
    args = _parse_args(argv)
    
    if args.command == "run":
        try:
            func = resolve_callable(args.target)
        except (ValueError, ModuleNotFoundError, AttributeError) as e:
            print(f"Error: {e}", file=sys.stderr)
            sys.exit(1)
        
        if args.verbose:
            print(f"Benchmarking {args.target} with sizes {args.sizes}", file=sys.stderr)
            print(f"  trials={args.trials}, warmup={args.warmup}", file=sys.stderr)
        
        analysis = benchmark_function(
            func,
            sizes=args.sizes,
            trials=args.trials,
            warmup=args.warmup,
            verbose=args.verbose,
            memory=args.memory,
        )
        
        if args.json:
            print(_analysis_to_json(analysis))
        else:
            _print_human(analysis, show_memory=args.memory)
        
        # Save baseline if requested
        if args.save_baseline:
            from .regression import save_baseline
            save_baseline(analysis, args.save_baseline, name=args.target)
            print(f"\nBaseline saved to: {args.save_baseline}")
        
        # Generate HTML report if requested
        if args.html:
            from .html_report import generate_html_report, save_html_report
            html_content = generate_html_report(analysis, title=f"Analysis: {args.target}")
            save_html_report(html_content, args.html)
            print(f"\nHTML report saved to: {args.html}")
        
        # Handle plotting
        if args.plot or args.plot_save:
            try:
                from .plotting import plot_analysis
                plot_analysis(
                    analysis,
                    title=f"Complexity Analysis: {args.target}",
                    save_path=args.plot_save,
                    show=args.plot and not args.plot_save,
                )
                if args.plot_save:
                    print(f"\nPlot saved to: {args.plot_save}")
            except ImportError as e:
                print(f"\nWarning: {e}", file=sys.stderr)
    
    elif args.command == "regression":
        try:
            func = resolve_callable(args.target)
        except (ValueError, ModuleNotFoundError, AttributeError) as e:
            print(f"Error: {e}", file=sys.stderr)
            sys.exit(1)
        
        from .regression import load_baseline, detect_regression
        
        try:
            baseline = load_baseline(args.baseline)
        except FileNotFoundError:
            print(f"Error: Baseline not found: {args.baseline}", file=sys.stderr)
            sys.exit(1)
        
        # Use baseline sizes if not specified
        sizes = args.sizes
        if sizes is None:
            sizes = [m["size"] for m in baseline.measurements]
        
        print(f"Running regression check against {args.baseline}...")
        print(f"  Baseline: {baseline.best_label}")
        
        analysis = benchmark_function(func, sizes=sizes, trials=3)
        
        result = detect_regression(analysis, baseline, time_threshold=args.threshold)
        
        print("\nResults:")
        print(f"  Current:  {analysis.best_label}")
        print(f"  Baseline: {baseline.best_label}")
        print(f"  Slowdown: {result.time_slowdown:.2f}x")
        print(f"\n{result.message}")
        
        if result.has_regression:
            print("\n❌ REGRESSION DETECTED")
            sys.exit(1)
        else:
            print("\n✅ No regression detected")
            sys.exit(0)
    
    elif args.command == "repl":
        from .interactive import start_repl
        start_repl()

    elif args.command == "explain":
        from .explanations import explain_complexity
        print(explain_complexity(args.complexity))

    elif args.command == "recommend":
        try:
            func = resolve_callable(args.target)
        except (ValueError, ModuleNotFoundError, AttributeError) as e:
            print(f"Error: {e}", file=sys.stderr)
            sys.exit(1)
        
        from .recommendations import suggest_sizes, format_recommendation
        
        print(f"Analyzing {args.target} for optimal sizes...")
        rec = suggest_sizes(func, time_budget=args.time_budget)
        print("\n" + format_recommendation(rec))

    elif args.command == "compare":
        targets = {}
        for t in args.targets:
            try:
                func = resolve_callable(t)
                targets[t] = func
            except (ValueError, ModuleNotFoundError, AttributeError) as e:
                print(f"Error resolving {t}: {e}", file=sys.stderr)
                sys.exit(1)
        
        from .multi_compare import compare_algorithms
        
        print(f"Comparing {len(targets)} algorithms on sizes {args.sizes}...", file=sys.stderr)
        result = compare_algorithms(targets, sizes=args.sizes, trials=args.trials)
        
        if args.json:
            # Simple JSON output
            data = {
                "winner": result.winner,
                "fastest": result.fastest,
                "results": [
                    {
                        "name": r.name,
                        "complexity": r.time_complexity,
                        "avg_time": r.avg_time,
                        "rank": r.rank
                    }
                    for r in result.results
                ]
            }
            print(json.dumps(data, indent=2))
        else:
            print("\n" + result.summary_table)

    elif args.command == "dashboard":
        targets = []
        for t in args.targets:
            try:
                func = resolve_callable(t)
                targets.append((t, func))
            except (ValueError, ModuleNotFoundError, AttributeError) as e:
                print(f"Error resolving {t}: {e}", file=sys.stderr)
                sys.exit(1)
        
        from .dashboard import generate_dashboard
        
        analyses = []
        print(f"Benchmarking {len(targets)} targets for dashboard...", file=sys.stderr)
        for name, func in targets:
            print(f"  Running {name}...", file=sys.stderr)
            analysis = benchmark_function(func, sizes=args.sizes)
            analysis.name = name  # Ensure name is set
            analyses.append(analysis)
            
        generate_dashboard(analyses, output_dir=args.output)
        print(f"\n✅ Dashboard generated at: {args.output}/index.html")

    elif args.command == "cloud":
        from .cloud import generate_github_action
        generate_github_action()


if __name__ == "__main__":
    main()
