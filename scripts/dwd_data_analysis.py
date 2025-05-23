import os

import pandas as pd
from ydata_profiling import ProfileReport


def analyze_data():
    df = pd.read_csv('../data/DWD.csv')
    report = ProfileReport(df, title="DWD Data Analysis")
    os.makedirs('../plots/dwd_analysis/', exist_ok=True)
    report.to_file('../plots/dwd_analysis/report.html')


def main():
    analyze_data()


if __name__ == '__main__':
    main()