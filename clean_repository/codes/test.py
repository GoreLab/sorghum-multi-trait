

import pandas as pd
import os
import numpy as np

os.chdir("/workdir/jp2476/transfer_data")
tmp = pd.read_csv('GBS.csv', index_col=0)

M.columns.isin(tmp.columns)

mask=tmp.columns.isin(M.columns)

tmp = tmp.transpose()[mask].transpose()


tmp.to_csv('gbs.csv')


tmp.index = M.index

test = M.loc[:, line_names] == tmp.loc[:, line_names]

test.values.sum()

test.shape[0] * test.shape[1]

