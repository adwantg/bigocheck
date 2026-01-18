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
""",
    )
    sub = parser.add_subparsers(dest="command", required=True)

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

    return parser.parse_args(argv)


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
        print(f"  {f.label:<12} error={f.error:.4f} scale={f.scale:.6g}{marker}")
    
    # Show space fits if available
    if show_memory and analysis.space_fits:
        print("\nSpace Fits (lower error is better):")
        for f in analysis.space_fits[:5]:  # Show top 5
            marker = " ★" if f.label == analysis.space_label else ""
            print(f"  {f.label:<12} error={f.error:.4f} scale={f.scale:.6g}{marker}")


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


if __name__ == "__main__":
    main()
