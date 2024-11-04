import gzip

def load_text(path, size_limit=None):
    with gzip.open(path, 'rt', encoding='latin-1') as f:
        if size_limit:
            return f.read(size_limit)  # Read up to `size_limit` characters
        return f.read()  # Read the full text if no limit is provided
