import argparse
import os

import pandas as pd
import numpy as np
from tqdm import tqdm

if __name__ == '__main__':
    args = argparse.ArgumentParser()
    args.add_argument('-i', '--input', help='Input file', required=True)
    args.add_argument('-o', '--output', help='Output path', required=True)

    args = args.parse_args()

    df = pd.read_csv(args.input, header=None)

    if not os.path.exists(args.output):
        os.makedirs(args.output)

    df = df.sort_values(by=[0])

    for i in tqdm(range(1, 41)):
        df_single = df[[0, i]]
        np.save(os.path.join(args.output, '{}.npy'.format(i-1)), df_single[i].to_numpy().reshape(-1, 1))
        df_single.to_csv(os.path.join(args.output, '{}.csv'.format(i-1)), index=False, header=False)
        # df_single.to_csv(os.path.join(args.output, '{}.csv'.format(i-1)), index=False, header=False)