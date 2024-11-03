def bwt_transform(text):
    rotations = sorted([text[i:] + text[:i] for i in range(len(text))])
    return "".join(row[-1] for row in rotations)

def inverse_bwt(bwt_text):
    table = [""] * len(bwt_text)
    for _ in range(len(bwt_text)):
        table = sorted([bwt_text[i] + table[i] for i in range(len(bwt_text))])
    return [row for row in table if row.endswith("$")][0][:-1]
