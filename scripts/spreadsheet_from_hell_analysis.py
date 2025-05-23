import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import pearsonr, spearmanr


def convert_to_float(x, default=-1.0):
    x = x.replace(' ', '')
    try:
        x = float(x)
    except:
        x = default

    return x

def valid_row(row):
    valid = True
    for v in row:
        if np.isnan(v) or v < 0:
            valid = False

    return valid

# def ef_qrs_corr(ef_col='EF_0', qrs_col='QRS (ms) 0'):
def ef_qrs_corr(ef_col='EF_8wks', qrs_col='QRS (ms) 8'):
    df = pd.read_excel('../data/spreadsheet_from_hell.xlsx')

    x_col = 'QRS'
    y_col = 'EF'
    values = {y_col: df[ef_col].apply(lambda x: convert_to_float(x) if isinstance(x, str) else x),
              x_col: df[qrs_col].apply(lambda x: convert_to_float(x) if isinstance(x, str) else x)}

    df_ef_qrs = pd.DataFrame(values)
    df_ef_qrs['valid'] = df_ef_qrs.apply(lambda row: valid_row(row), axis=1)
    df_ef_qrs = df_ef_qrs[df_ef_qrs['valid'] == True]

    # Derive values and create scatter-plot
    x, y = df_ef_qrs[x_col].values, df_ef_qrs[y_col].values
    plt.scatter(x, y, color='red')
    print('{} mean={:.2f} stddev={:.2f}'.format(x_col, np.mean(x), np.std(x)))
    print('{} mean={:.2f} stddev={:.2f}'.format(y_col, np.mean(y), np.std(y)))

    # Perform least squares polynomial fit to calculate regression line
    m, b = np.polyfit(x, y, 1)
    plt.plot(x, m * x + b, color='blue')

    # Calculate correlation coefficients
    pcc, pp = pearsonr(x, y)
    scc, sp = spearmanr(x, y)

    # Annotate with PCC and SCC
    annotation = f"PCC={pcc:.2f} (p={pp:.3f})\nSCC={scc:.2f} (p={sp:.3f})"
    plt.text(0.05, 0.95, annotation, transform=plt.gca().transAxes, verticalalignment='top', fontsize=10, bbox=dict(facecolor='white', alpha=0.7, edgecolor='gray'))
    plt.xlabel(x_col)
    plt.ylabel(y_col)
    plt.grid(True)
    plt.show()


if __name__ == '__main__':
    ef_qrs_corr()