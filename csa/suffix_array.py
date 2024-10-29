def build_suffix_array(text):
    suffixes = sorted((text[i:], i) for i in range(len(text)))
    return [suffix[1] for suffix in suffixes]
