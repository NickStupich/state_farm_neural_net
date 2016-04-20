import numpy as np
import pandas


fn = 'driver_imgs_list.csv'
pd = pandas.read_csv(fn)

uniqueSubjects = list(set(pd["subject"]))

print uniqueSubjects