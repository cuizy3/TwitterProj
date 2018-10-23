# -*- coding: utf-8 -*-
"""
Created on Mon Oct 22 19:59:43 2018

@author: Eddie
"""

import load_embeddings
import numpy as np

embedding = load_embeddings.load_vectors('embeddings/wiki.en.vec')

for key in embedding:
    print key
    
fr0m = embedding['from']

print (fr0m)
