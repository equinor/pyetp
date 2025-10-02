import random
import string
from itertools import islice
from typing import Iterable
def short_id(length=8):
    return ''.join([random.choice(string.ascii_letters + string.digits + '-_') for _ in range(length)])

def batched(iterable:Iterable, n: int):
    "Batch data into tuples of length n. The last batch may be shorter."
    # batched('ABCDEFG', 3) --> ABC DEF G
    if n < 1:
        raise ValueError('n must be at least one')
    it = iter(iterable)
    while batch := tuple(islice(it, n)):
        yield batch