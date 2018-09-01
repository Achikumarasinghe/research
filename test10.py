import os
os.environ['R_USER'] = 'C:/Users/USER/AppData/Local/Programs/Python/Python36/Lib/site-packages/rpy2'
import rpy2.robjects as robjects
r=robjects.r
import subprocess
from numpy import *
import scipy as sp
import pandas as pd
from rpy2.robjects.packages import importr
import rpy2.robjects as ro
from rpy2.robjects import pandas2ri
pandas2ri.activate()

df = pd.DataFrame({'A': [1, 2, 3], 'B': [4, 5, 6], 'C':[7,8,9]})
df.to_csv("dataframe.csv",index=False)
r_dataframe = pandas2ri.py2ri(df)
##print(type(r_dataframe))
print(r_dataframe)

command = 'C:/Program Files/R/R-3.5.1/bin/Rscript'
path2script = 'E:/SLIIT1/4th year/RESEARCH/Workspace/research/testr.R'
args = r_dataframe
retcode = subprocess.call([command, path2script], shell=True)