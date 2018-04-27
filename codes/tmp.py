# Yield successive n-sized chunks from x
def chunks(x, n):
    for i in range(0, len(x), n):
        yield x[i:i + n]