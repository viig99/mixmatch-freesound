# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import librosa
from fastai.basics import *

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory
import random
import os

random.seed(9)


sample_submission_file = '../input/sample_submission.csv'
header_line = open(sample_submission_file).readlines()[0].strip()
print(header_line)

def get_random_prediction():
    random_weights = [random.randint(1,100) for item in range(80)]
    denominator = sum(random_weights)
    random_probs = [(w*1.0)/denominator for w in random_weights]
    return random_probs
    
#print(os.listdir("../input/test"))
ofp = open('submission.csv', 'w')

ofp.write(header_line)
ofp.write('\n')

for test_file in os.listdir("../input/test"):
  rand_pred_list = get_random_prediction()
  output_cols = [test_file] + rand_pred_list
  output_line = ','.join([str(col) for col in output_cols])
  ofp.write(output_line)
  ofp.write('\n')
print('H: finished...')
