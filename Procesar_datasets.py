#!/usr/bin/env python
# coding: utf-8

# In[35]:


import pandas as pd
import glob
import os
import json
from pandas.io.json import json_normalize


# In[50]:


def join_dataset():
    list_model = ['rf', 'lr', 'svc']

    join_dir= "./join_model/"

    lista_train =[]
    lista_test =[]
    lista_json=[]
    for l in list_model:
        try:
          os.stat(join_dir)
        except:
          os.mkdir(join_dir)
        path_train= './' + l +"_res/*_train.csv"
        path_test= './' + l +"_res/*_test.csv"
        path_json= './new_expe/' + "PARAMETERS" +"*_NE.json"
        allFiles_train = glob.glob(path_train)
        allFiles_test = glob.glob(path_test)
        allFiles_json = glob.glob(path_json)
        '''for file in allFiles_train:
            print("f_train: ", file)
            df = pd.read_csv(file,index_col=None, header=0,engine='python')
            lista_train.append(df)  
        print("")
        for file in allFiles_test:
            print("f_test: ", file)
            df = pd.read_csv(file,index_col=None, header=0,engine='python')
            lista_test.append(df)  '''
        for file in allFiles_json:
            print("f_json: ", file)
            df = json.loads(open(file).read())
            lista_json.append(df)  
        print("")
              
        '''file_output_train = "all_" + l + "_train" + ".csv"
        data=pd.concat(lista_train)
        data.to_csv(join_dir + file_output_train)
        
        file_output_test = "all_" + l + "_test" + ".csv"
        data=pd.concat(lista_test)
        data.to_csv(join_dir + file_output_test)'''
        
        file_output_json = "all_PARAMETERS_" + ".csv"
        dd =pd.DataFrame(lista_json)
        
        
        # data.to_csv(join_dir + file_output_json)
        
        lista_train=[]
        lista_test =[]
        lista_json=[]
    print("Join finished. Ok¡¡")
    print(dd)
    # dd=dd.Series(dd)
    print(dd.loc[:,'max_features'].value_counts())
        


# In[51]:



join_dataset()

