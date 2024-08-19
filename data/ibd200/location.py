def target(meta):
    name = 'IBD200: CD location (ileum, colon or both)'
    cats = {'ileal': 0, 'colonic': 1, 'ileocolonic': 2}
    prop = meta.query('diagnosis == "cd"')['location'].map(
        cats).dropna().astype(int)
    return prop, name
