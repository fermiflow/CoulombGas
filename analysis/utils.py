import re 

def parse_filename(f):
    n = int(re.search('n_([0-9]*)_', f).group(1))
    dim = int(re.search('dim_([0-9]*)_', f).group(1))
    rs = float(re.search('rs_([0-9]*\.?[0-9]*)_', f).group(1))
    T = float(re.search('Theta_([0-9]*\.?[0-9]*)_', f).group(1))

    d = int(re.search('depth_([0-9]*)_', f).group(1))
    h1 = int(re.search('spsize_([0-9]*)_', f).group(1))
    h2 = int(re.search('tpsize_([0-9]*)_', f).group(1))

    #b = int(re.search('b([0-9]*)_', f).group(1))
    b = None

    return n, dim, rs, T, d, h1, h2, b


def natural_sort(l):
    convert = lambda text: int(text) if text.isdigit() else text.lower()
    alphanum_key = lambda key: [ convert(c) for c in re.split('([0-9]+)', key) ]
    return sorted(l, key = alphanum_key)
