# -*- coding: utf-8 -*-
"""
Created on Sat Jul 14 16:52:04 2018

@author: Lucilla Chen
"""
#%%
import pandas as pd
import numpy as np
from datetime import datetime
from datetime import timedelta
from queue import Queue
import os
global rootpath
rootpath = 'C:\\Users\\ChengZeYang\\Desktop\\CTApair\\'
os.chdir(rootpath)
import importlib
import Func_CTA_pair 
import matplotlib.pyplot as plt
importlib.reload(Func_CTA_pair)
import CTAStrategy
importlib.reload(CTAStrategy)
global outfile
outfile = "output\\"

#%%一键测试因子
class ParaPerformDirect():
    StrategyType = 'pair'
    DataPath = "D:\\ComMin05\\"
    outpath = rootpath + outfile
    strategylist = [40]
    factortypelist = pd.read_excel(rootpath + "factor_item_type.xlsx") 
    startDate = '2013-12-31 16:00:00'
    endDate = '2021-07-31 16:00:00'
    fee = 0.0001
    slippage = 0.0002
    tswindow = 3000
    group = 11
    ifDelHolidayRet = True
    if_add_para = False

paraperform = ParaPerformDirect()
Result = Func_CTA_pair.Strategy_pair(paraperform)


#%%初步判断合适的品种
itemneed =['I_J','I_RB','TA_MA']
itemlen = len(itemneed)
No = 11 #因子编号
items_for_strategy = pd.DataFrame()
all_items_select = set([])
print ("因子",No)
performance_combine_all = pd.DataFrame()
useful_itemset = set([])
df_perform = pd.read_excel(paraperform.outpath + "perform_combine\\Strategy" + str(No) + "_PerformanceAll.xlsx")
parameters = {}
parameters['a'] = str(paraperform.factortypelist.loc[paraperform.factortypelist['factor'] == No, 'paras_a'].values[0]).split(',')
parameters['b'] = str(paraperform.factortypelist.loc[paraperform.factortypelist['factor'] == No, 'paras_b'].values[0]).split(',')
paraRows = Func_CTA_pair.iterdict({'a':parameters['a'], 'b':parameters['b']})
for parameter in paraRows:
    c = "factor" + str(No) + "_" + str(parameter['a']) + '_' + str(parameter['b']) 
    df = pd.read_csv(paraperform.outpath + "dailyretbase\\" + c + ".csv")
    df['Ret'] = df[itemneed].mean(axis = 1)
    df = df.sort_values('Date', ascending = True).reset_index(drop = True)
    df[c] = df['Ret']
    performance_combine = Func_CTA_pair.getPerformance_universe(df[['Date','Ret']])
    performance_combine['factor'] = c  
    performance_combine['items'] = ','.join(itemneed)
    performance_combine_all = pd.concat([performance_combine_all, performance_combine], axis = 0)   
performance_combine_yearall = performance_combine_all.loc[performance_combine_all['year'] == 'All',:]
items = performance_combine_yearall.loc[performance_combine_yearall['卡玛']== performance_combine_yearall['卡玛'].max(), 'items'].values[0].split('_')
print (items)
all_items_select = all_items_select.union(set(items))
','.join(itemneed)  