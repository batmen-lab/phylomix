def target(meta):
    name = 'Alzbiom: health vs Alzheimer\'s disease'
    prop = meta['AD'].astype(int)
    return prop, name
