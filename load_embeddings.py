# -*- coding: utf-8 -*-
"""
Created on Mon Oct 22 19:58:59 2018

@author: Eddie
"""

import io

def load_vectors(fname):
    fin = io.open(fname, 'r', encoding='utf-8', newline='\n', errors='ignore')
    n, d = map(int, fin.readline().split())
    data = {}
    count = 0
    for line in fin:
        tokens = line.rstrip().split(' ')
        data[tokens[0]] = map(float, tokens[1:])
        count += 1
        if count == 25:
            break
    return data