# -*- coding: utf-8 -*-
"""
Created on Tue Jan 21 15:43:14 2020

@author: KyleCheng
"""

import pandas as pd
import openpyxl

strategy_num = "24"
file = "V:\\CTApair\\output\\Strategy%s_PerformanceByItem.xlsx" % strategy_num
sheets = openpyxl.load_workbook(file).sheetnames

good_paras_all = pd.DataFrame()

for sheet in sheets:
    data = pd.read_excel(file, sheetname = sheet)
    good_paras = data.loc[(data.year == "All") & (data["夏普"] > 0.3) & (data["卡玛"] > 0.3) & (data['总换手']>=20)]
    if good_paras.shape[0] > 0:
        good_paras_all = good_paras_all.append(good_paras)
    print(sheet + " done")

good_rtn_item = pd.DataFrame()
item_list = []
for i in good_paras_all.groupby(["item"]):
    if len(i[1]) >= 2:
        good_rtn_item = pd.concat([good_rtn_item, i[1]], axis=0, ignore_index=True)    
        item_list.append(i[0])

'''
good_rtn_para = pd.DataFrame()
for i in good_paras_all.groupby(["a",'b']):
    if len(i[1]) >= 3:
        good_rtn_para = pd.concat([good_rtn_para, i[1]], axis=0, ignore_index=True)
'''  
