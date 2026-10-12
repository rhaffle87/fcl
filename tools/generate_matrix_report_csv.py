# tools/generate_matrix_report_csv.py — Extract all 72 matrix combinations into a standardized benchmark CSV via SSH/MLflow.
# Compiles evaluation data into a single master summary spreadsheet.

import argparse
import os
import subprocess

import pandas as pd


def main():
    py_code = """
import sqlite3
import pandas as pd

conn = sqlite3.connect('/root/mlflow.db')

query = '''
SELECT r.run_uuid, r.experiment_id, r.status, r.start_time, r.end_time,
       p_model.value AS model_type,
       p_cl.value AS cl_strategy,
       p_agg.value AS aggregation_strategy,
       p_dp.value AS dp_enabled,
       p_lr.value AS learning_rate,
       p_bs.value AS batch_size
FROM runs r
JOIN params p_model ON r.run_uuid = p_model.run_uuid AND p_model.key = 'model_type'
JOIN params p_cl ON r.run_uuid = p_cl.run_uuid AND p_cl.key = 'cl_strategy'
JOIN params p_agg ON r.run_uuid = p_agg.run_uuid AND p_agg.key = 'aggregation_strategy'
LEFT JOIN params p_dp ON r.run_uuid = p_dp.run_uuid AND p_dp.key = 'dp_enabled'
LEFT JOIN params p_lr ON r.run_uuid = p_lr.run_uuid AND p_lr.key = 'lr'
LEFT JOIN params p_bs ON r.run_uuid = p_bs.run_uuid AND p_bs.key = 'batch_size'
WHERE r.status = 'FINISHED'
'''
df_runs = pd.read_sql_query(query, conn)

metrics_query = 'SELECT run_uuid, key, value, step FROM metrics'
df_m = pd.read_sql_query(metrics_query, conn)
latest = df_m.sort_values('step').groupby(['run_uuid', 'key']).last().reset_index()
pivoted = latest.pivot(index='run_uuid', columns='key', values='value').reset_index()

df = pd.merge(df_runs, pivoted, on='run_uuid', how='inner')
df['dp_enabled'] = df['dp_enabled'].fillna('false').astype(str).str.lower()

if 'final_best_loss' in df.columns:
    df['loss'] = df['loss'].fillna(df['final_best_loss'])
if 'best_loss' in df.columns:
    df['loss'] = df['loss'].fillna(df['best_loss'])
if 'final_best_accuracy' in df.columns:
    df['accuracy'] = df['accuracy'].fillna(df['final_best_accuracy'])

size_map = {'cnn': 46447, 'mlp': 19843, 'transformer': 74240}
df['quantized_bytes'] = df['model_type'].map(size_map)

cols_order = [
    'run_uuid', 'model_type', 'cl_strategy', 'aggregation_strategy', 'dp_enabled',
    'accuracy', 'loss', 'crucial_model_performance', 'quantized_bytes', 'communication_bytes',
    'accuracy_class_0', 'accuracy_class_1', 'accuracy_class_2', 'accuracy_class_3', 'accuracy_class_4',
    'f1_class_0', 'f1_class_1', 'f1_class_2', 'f1_class_3', 'f1_class_4'
]
avail_cols = [c for c in cols_order if c in df.columns]
print(df[avail_cols].to_csv(index=False))
"""

    out_csv = "data/reports/benchmarks/master_matrix_benchmark_report.csv"
    os.makedirs(os.path.dirname(out_csv), exist_ok=True)

    if args.local or not os.path.exists("/root/mlflow.db"):
        local_db = args.db_path or "data/db/mlflow.db"
        if os.path.exists(local_db):
            print(f"[*] Extracting directly from local database: {local_db}")
            local_py = py_code.replace("conn = sqlite3.connect('/root/mlflow.db')", f"conn = sqlite3.connect(r'{local_db}')")
            res = subprocess.run([sys.executable, "-c", local_py], capture_output=True, text=True, encoding="utf-8")
            if res.returncode == 0:
                with open(out_csv, "w", encoding="utf-8") as f:
                    f.write(res.stdout)
                df_check = pd.read_csv(out_csv)
                print(f"[SUCCESS] Exported {len(df_check)} matrix run records to {out_csv}")
                return

    cmd = [
        "ssh",
        "-o",
        "BatchMode=yes",
        "-o",
        "StrictHostKeyChecking=no",
        "root@10.10.130.10",
        "/opt/flower-env/bin/python3",
        "-",
    ]
    res = subprocess.run(
        cmd, input=py_code, capture_output=True, text=True, encoding="utf-8"
    )
    if res.returncode != 0:
        print("ERROR:", res.stderr)
        return

    with open(out_csv, "w", encoding="utf-8") as f:
        f.write(res.stdout)

    df_check = pd.read_csv(out_csv)
    print(f"[SUCCESS] Exported {len(df_check)} matrix run records to {out_csv}")


if __name__ == "__main__":
    import sys
    parser = argparse.ArgumentParser(
        description="72-Combination Benchmark Matrix CSV Extractor"
    )
    parser.add_argument("--local", action="store_true", help="Extract from local SQLite database")
    parser.add_argument("--db-path", default=None, help="Custom path to SQLite database")
    args = parser.parse_args()
    main()
