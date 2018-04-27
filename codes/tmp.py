


X['height'] = pd.concat([X['height'], tmp.dot(W_bin.loc[tmp.columns.tolist()])], axis=1)

X['height'] = np.hstack((np.dot(tmp, W_bin.loc[tmp.columns.tolist()]), X['height']))


tmp1 = pd.concat([X['height'], tmp.dot(W_bin.loc[tmp.columns.tolist()])], axis=1)

tmp2 = np.hstack((np.dot(tmp, W_bin.loc[tmp.columns.tolist()]), X['height']))

tmp1.values == tmp2