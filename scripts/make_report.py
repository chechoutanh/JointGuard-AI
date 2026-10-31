"""Write the report (Markdown + Word) from saved results, in Vietnamese (default) or English.

  python -m scripts.make_report --results results/standard --model models/standard --out reports/standard
  python -m scripts.make_report --results results/standard --model models/standard --out reports/standard --lang en
"""
import argparse
from pathlib import Path

from src.report import write_report
from src.report_en import write_report_en


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", required=True)
    ap.add_argument("--model", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--lang", choices=["vi", "en"], default="vi")
    ap.add_argument("--overwrite", action="store_true")
    a = ap.parse_args(argv)
    name = f"report_{a.lang}.md"
    if (Path(a.out) / name).exists() and not a.overwrite:
        raise SystemExit(f"{a.out}/{name} exists; use --overwrite to replace it.")
    md, docx = (write_report_en if a.lang == "en" else write_report)(a.results, a.model, a.out)
    print("saved", md)
    print("saved", docx)


if __name__ == "__main__":
    main()
