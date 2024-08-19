def target(meta):
    name = 'IBD200: IBD type (UC vs CD)'
    cats = {'uc': 0, 'cd': 1}
    prop = meta['diagnosis'].map(cats).dropna().astype(int)
    return prop, name
