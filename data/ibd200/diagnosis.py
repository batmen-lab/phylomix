def target(meta):
    name = 'IBD200: diagnosis (Healthy vs UC vs CD)'
    cats = {'healthy_control': 0, 'uc': 1, 'cd': 2}
    prop = meta['diagnosis'].map(cats).dropna().astype(int)
    return prop, name
