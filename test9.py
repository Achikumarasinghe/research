import os
os.environ['R_USER'] = 'C:/Users/USER/AppData/Local/Programs/Python/Python36/Lib/site-packages/rpy2'
from numpy import *
import scipy as sp
import pandas as pd
from rpy2.robjects.packages import importr
import rpy2.robjects as ro
from rpy2.robjects import pandas2ri
pandas2ri.activate()

#nfca = importr("nFCA")
#utils = importr("utils")
#nfca_example = utils.data("nfca_example", package = "nFCA")
#print(nfca_example)

df = pd.DataFrame({'A': [1, 2, 3], 'B': [4, 5, 6], 'C':[7,8,9]},index=["one", "two", "three"])
r_dataframe = pandas2ri.py2ri(df)
##print(type(r_dataframe))
print(r_dataframe)

#nfca = importr("nFCA")
#nfca(data = r_dataframe)




