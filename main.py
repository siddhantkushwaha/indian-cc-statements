import os
import json
import argparse

import pandas as pd

from indian_cc_statements.parser import extract


def main():
    args = argparse.ArgumentParser()
    args.add_argument(
        "--pdf-path",
        type=str,
        required=True,
        help="Comma separated list of PDF paths",
    )
    args.add_argument(
        "--password",
        type=str,
        required=False,
        help="Comma separated list of passwords",
    )
    args = args.parse_args()
    pdf_paths = args.pdf_path.split(",")
    passwords = args.password.split(",") if args.password else []

    for pdf_path in pdf_paths:
        pdf_name = os.path.basename(pdf_path)
        print(pdf_path)
        result = extract(
            pdf_path,
            expand_x=0.15,
            expand_y=0.10,
            confidence_threshold=0.5,
            passwords=passwords,
        )

        if result:
            data_dir = os.path.join("statement_summaries", pdf_name)
            os.makedirs(data_dir, exist_ok=True)

            # Save consolidated results as json
            with open(os.path.join(data_dir, "summary.json"), "w") as f:
                json.dump(result, f, indent=4)

            # Save consolidated results as csv
            df = pd.DataFrame(result)
            df.to_csv(os.path.join(data_dir, "summary.csv"), index=False)
        else:
            print(f"No transactions found in {pdf_path}")


if __name__ == "__main__":
    main()
