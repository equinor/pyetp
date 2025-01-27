import random
import string


def short_id(length=8):
    return ''.join([random.choice(string.ascii_letters + string.digits + '-_') for _ in range(length)])
