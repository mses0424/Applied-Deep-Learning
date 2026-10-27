"""Evidence-backed v3 report; qualitative interpretation must be completed by the student."""
import sys
from build_report_v2 import main

if __name__ == "__main__":
    sys.argv[1:1] = ["--v3", "--source", "reports/v3/report_source.md", "--output", "reports/v3/report",
                    "--evaluation", "results/v3/evaluation.json", "--generations", "results/v3/generations.json",
                    "--predictions", "results/v3/public.json"]
    main()
