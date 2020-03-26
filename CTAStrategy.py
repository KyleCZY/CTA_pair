# -*- coding: utf-8 -*-
"""
Created on Tue Oct 30 18:15:51 2018

@author: Administrator
"""

import pandas as pd
import numpy as np
global rootpath
import importlib
import Func_CTA_pair
importlib.reload(Func_CTA_pair)
import math
import talib
#Data为行情数据。类型为DataFrame，对应字段'ContractID', 'time', 'open_adj', 'high_adj', 
#'low_adj', 'cp_adj', 'vwap_adj', 's1_adj', 'b1_adj','mid_adj', 'volume', 
#'turnover'(其中mid_adj = 0.5*(s1_adj + b1_adj))
#parameter为字典，参数值通过字典对应的key获取
cut = 1500


def smoothfactor(factor, cut, b):
    factor = factor.ewm(halflife=b, min_periods=cut, adjust=True, ignore_na=True).mean()
    return factor    


#收益率差的指数加权移动平均,动量
def factorMaker1(Data, parameter, pair):
    a = int(parameter['a'])
    b = int(parameter['b'])
    df = Data.copy()
    df['tmp1'] = df['cp_adj_' + pair[0]].pct_change(a)
    df['tmp2'] = df['cp_adj_' + pair[1]].pct_change(a)
    factor = df['tmp1'] - df['tmp2']
    factor = smoothfactor(factor, cut, b)
    factor = np.sign(factor)
    return factor

#标准差*收益率
def factorMaker2(Data, parameter, pair):
    a = int(parameter['a'])
    b = int(parameter['b'])
    df = Data.copy()
    df['tmp1'] = df['cp_adj_' + pair[0]].pct_change(a)
    df['tmp2'] = df['cp_adj_' + pair[1]].pct_change(a)
    df['std1'] = df['cp_adj_' + pair[0]].pct_change().rolling(a).std()
    df['std2'] = df['cp_adj_' + pair[1]].pct_change().rolling(a).std() 
    factor = df['tmp1']*df['std1'] - df['tmp2']*df['std2']
    factor = smoothfactor(factor, cut, b)
    factor = np.sign(factor)
    return factor    

    
def factorMaker3(Data, parameter, pair):
    a = int(parameter['a'])
    b = int(parameter['b'])
    df = Data.copy()
    df['tmp1'] = df['high_adj_' + pair[0]].pct_change(a)
    df['tmp2'] = df['high_adj_' + pair[1]].pct_change(a)
    factor = df['tmp1'] - df['tmp2']
    factor = smoothfactor(factor, cut, b)
    factor = np.sign(factor)
    return factor

#对数价差策略（中信研报20170516）
#动量
def factorMaker4(Data, parameter, pair):
    a = int(parameter['a'])
    b = int(parameter['b'])
    df = Data.copy()
    df['cp_adj_' + pair[0]] = df['cp_adj_' + pair[0]].map(lambda x : math.log(x))
    df['cp_adj_' + pair[1]] = df['cp_adj_' + pair[1]].map(lambda x : math.log(x))
    df['xy'] = df['cp_adj_' + pair[0]] * df['cp_adj_' + pair[1]]
    df["sumxy"] = df['xy'].rolling(a).sum()
    df["meanx"] = df['cp_adj_' + pair[0]].rolling(a).mean()
    df["meany"] = df['cp_adj_' + pair[1]].rolling(a).mean()
    df['xx'] = df['cp_adj_' + pair[0]] * df['cp_adj_' + pair[0]]
    df["sumxx"] = df['xx'].rolling(a).sum()
    df["b1"] = (df["sumxy"] - a * df["meanx"] * df["meany"]) / \
               (df["sumxx"] - a * df["meanx"] * df["meanx"])
    df["b0"] = df["meany"] - df["meanx"] * df["b1"]
    df["b0_est"] = df['cp_adj_' + pair[1]] - df['cp_adj_' + pair[0]] * df["b1"].shift(1)
    df["factor"] = df["b0_est"] - df["b0"].shift(1)
    factor = df["factor"]
    factor = smoothfactor(factor, cut, b)
    factor = np.sign(factor) 
    return factor    

#反转
def factorMaker5(Data, parameter, pair):
    a = int(parameter['a'])
    b = int(parameter['b'])
    df = Data.copy()
    df['cp_adj_' + pair[0]] = df['cp_adj_' + pair[0]].map(lambda x : math.log(x))
    df['cp_adj_' + pair[1]] = df['cp_adj_' + pair[1]].map(lambda x : math.log(x))
    df['xy'] = df['cp_adj_' + pair[0]] * df['cp_adj_' + pair[1]]
    df["sumxy"] = df['xy'].rolling(a).sum()
    df["meanx"] = df['cp_adj_' + pair[0]].rolling(a).mean()
    df["meany"] = df['cp_adj_' + pair[1]].rolling(a).mean()
    df['xx'] = df['cp_adj_' + pair[0]] * df['cp_adj_' + pair[0]]
    df["sumxx"] = df['xx'].rolling(a).sum()
    df["b1"] = (df["sumxy"] - a * df["meanx"] * df["meany"]) / \
               (df["sumxx"] - a * df["meanx"] * df["meanx"])
    df["b0"] = df["meany"] - df["meanx"] * df["b1"]
    df["b0_est"] = df['cp_adj_' + pair[1]] - df['cp_adj_' + pair[0]] * df["b1"].shift(1)
    df["error"] = df["b0_est"] - df["b0"].shift(1)
    factor = df["error"]
    factor = smoothfactor(factor, cut, b)
    factor = np.sign(factor)
    return factor    

def factorMaker6(Data, parameter, pair):
    a = int(parameter['a'])
    b = int(parameter['b'])
    df = Data.copy()
    df['tmp1'] = df['cp_adj_' + pair[0]].pct_change(a)
    df['tmp2'] = df['cp_adj_' + pair[1]].pct_change(a)
    factor = df['tmp1'] - df['tmp2']
    factor = smoothfactor(factor, cut, b)
    return factor

def factorMaker7(Data, parameter, pair):
    a = int(parameter['a'])
    b = int(parameter['b'])
    df = Data.copy()
    df['tmp1'] = df['cp_adj_' + pair[0]].pct_change(a)
    df['tmp2'] = df['cp_adj_' + pair[1]].pct_change(a)
    factor = df['tmp2'] - df['tmp1']
    factor = smoothfactor(factor, cut, b)
    return factor

def factorMaker8(Data, parameter, pair):
    a = int(parameter['a']) 
    b = int(parameter['b'])
    df = Data.copy()
    df['tmp1'] = df['high_adj_' + pair[0]].pct_change(a)
    df['tmp2'] = df['high_adj_' + pair[1]].pct_change(a)
    factor = df['tmp1'] - df['tmp2']
    factor = smoothfactor(factor, cut, b)
    factor = np.sign(factor)
    return factor

def factorMaker9(Data, parameter, pair):
    a = int(parameter['a'])
    b = int(parameter['b'])
    df = Data.copy()
    df['tmp1'] = df['high_adj_' + pair[0]].pct_change(a)
    df['tmp2'] = df['high_adj_' + pair[1]].pct_change(a)
    factor = df['tmp2'] - df['tmp1']
    factor = smoothfactor(factor, cut, b)
    factor = np.sign(factor)
    return factor


#Acceleration Bands (ABANDS) 1 动量
def factorMaker10(Data, parameter, pair):
    a = int(parameter['a'])
    b = int(parameter['b'])
    df = Data.copy()
    df['tmp'] = df['cp_adj_' + pair[0]] / df['cp_adj_' + pair[1]]
    df['low'] =  df['tmp'].rolling(a).min()
    df['high'] = df['tmp'].rolling(a).max()
    df['upper'] = df['high'] * (1 + 3 * (df['high'] - df['low']) / ((df['high'] + df['low'])))
    df['down'] = df['low'] * (1 - 3 * (df['high'] - df['low']) / ((df['high'] + df['low'])))
    df['upband'] = df['upper'].rolling(b).mean()
    df['downband'] = df['down'].rolling(b).mean()
    factor = pd.Series(np.zeros(len(df['tmp'])))
    factor[(df['tmp']>df['upband'])] = 1
    factor[(df['tmp']<df['downband'])] = -1
    return factor

#Acceleration Bands (ABANDS) 2 反转
def factorMaker11(Data, parameter, pair):
    a = int(parameter['a'])
    b = int(parameter['b'])
    df = Data.copy()
    df['tmp'] = df['cp_adj_' + pair[0]] / df['cp_adj_' + pair[1]]
    df['low'] =  df['tmp'].rolling(a).min()
    df['high'] = df['tmp'].rolling(a).max()
    df['upper'] = df['high'] * (1 + 3 * (df['high'] - df['low']) / ((df['high'] + df['low'])))
    df['down'] = df['low'] * (1 - 3 * (df['high'] - df['low']) / ((df['high'] + df['low'])))
    df['upband'] = df['upper'].rolling(b).mean()
    df['downband'] = df['down'].rolling(b).mean()
    factor = pd.Series(np.full(len(df['tmp']), np.nan))
    factor[(df['tmp']>df['upband'])] = -1
    factor[(df['tmp']<df['downband'])] = 1
    factor = factor.fillna(method = 'ffill')
    return factor

#Absolute Price Oscillator (APO) 1 动量
def factorMaker12(Data, parameter, pair):
    a = int(parameter['a'])
    b = int(parameter['b'])
    df = Data.copy()
    df['tmp'] = df['cp_adj_' + pair[0]] / df['cp_adj_' + pair[1]]
    df['tmp1'] = df['tmp'].ewm(halflife = a).mean()
    df['tmp2'] = df['tmp'].ewm(halflife = b).mean()
    df['apo'] = df['tmp1'] - df['tmp2']
    factor = pd.Series(np.zeros(len(df['tmp'])))
    factor[(df['apo']>0)] = 1
    factor[(df['tmp']<0)] = -1
    return factor

#Absolute Price Oscillator (APO) 2 反转
def factorMaker13(Data, parameter, pair):
    a = int(parameter['a'])
    b = int(parameter['b'])
    df = Data.copy()
    df['tmp'] = df['cp_adj_' + pair[0]] / df['cp_adj_' + pair[1]]
    df['tmp1'] = df['tmp'].ewm(halflife = a).mean()
    df['tmp2'] = df['tmp'].ewm(halflife = b).mean()
    df['apo'] = df['tmp1'] - df['tmp2']
    factor = pd.Series(np.zeros(len(df['tmp'])))
    factor[(df['tmp']>0)] = -1
    factor[(df['tmp']<0)] = 1
    return factor

#Aroon (AR) 1
def factorMaker14(Data, parameter, pair):
    a = int(parameter['a'])
    b = int(parameter['b'])
    df = Data.copy()
    df['tmp'] = df['cp_adj_' + pair[0]] / df['cp_adj_' + pair[1]]
    df['tmp'] = df['tmp'].rolling(a).mean()
    df['up'],df['down'] = talib.AROON(df['tmp'].values,df['tmp'].values,b)
    factor = pd.Series(np.zeros(len(df['tmp'])))
    factor[(df['up'] > df['down']) & (df['up'] > 50)] = 1
    factor[(df['up'] < df['down']) & (df['down'] > 50)] = -1
    return factor

#Aroon (AR) 2
def factorMaker15(Data, parameter, pair):
    a = int(parameter['a'])
    b = int(parameter['b'])
    df = Data.copy()
    df['tmp'] = df['cp_adj_' + pair[0]] / df['cp_adj_' + pair[1]]
    df['tmp'] = df['tmp'].rolling(a).mean()
    df['up'],df['down'] = talib.AROON(df['tmp'].values,df['tmp'].values,b)
    factor = pd.Series(np.zeros(len(df['tmp'])))
    factor[(df['up'] > df['down']) & (df['up'] > 50)] = -1
    factor[(df['up'] < df['down']) & (df['down'] > 50)] = 1
    return factor

#Aroon Oscillator (ARO) 1
def factorMaker16(Data, parameter, pair):
    a = int(parameter['a'])
    b = int(parameter['b'])
    df = Data.copy()
    df['tmp'] = df['cp_adj_' + pair[0]] / df['cp_adj_' + pair[1]]
    df['tmp'] = df['tmp'].rolling(a).mean()
    df['up'],df['down'] = talib.AROON(df['tmp'].values,df['tmp'].values,b)
    factor = pd.Series(np.zeros(len(df['tmp'])))
    factor[(df['up'] > df['down'])] = -1
    factor[(df['up'] < df['down'])] = 1
    return factor

#Aroon Oscillator (ARO) 2
def factorMaker17(Data, parameter, pair):
    a = int(parameter['a'])
    b = int(parameter['b'])
    df = Data.copy()
    df['tmp'] = df['cp_adj_' + pair[0]] / df['cp_adj_' + pair[1]]
    df['tmp'] = df['tmp'].rolling(a).mean()
    df['up'],df['down'] = talib.AROON(df['tmp'].values,df['tmp'].values,b)
    factor = pd.Series(np.zeros(len(df['tmp'])))
    factor[(df['up'] > df['down'])] = 1
    factor[(df['up'] < df['down'])] = -1
    return factor

#Bollingger Band (BBANDS) 1 动量
def factorMaker18(Data, parameter, pair):
    a = int(parameter['a'])
    b = int(parameter['b'])
    df = Data.copy()
    df['tmp'] = df['cp_adj_' + pair[0]] / df['cp_adj_' + pair[1]]
    df['ma'] = df['tmp'].rolling(a).mean()
    df['std'] = df['tmp'].rolling(b).std()
    df['up'] = df['ma'] + df['std']
    df['down'] = df['ma'] - df['std']
    factor = pd.Series(np.zeros(len(df['tmp'])))
    factor[(df['tmp']>df['up'])] = 1
    factor[(df['tmp']<df['down'])] = -1
    return factor

#Bollingger Band (BBANDS) 2 反转
def factorMaker19(Data, parameter, pair):
    a = int(parameter['a'])
    b = int(parameter['b'])
    df = Data.copy()
    df['tmp'] = df['cp_adj_' + pair[0]] / df['cp_adj_' + pair[1]]
    df['ma'] = df['tmp'].rolling(a).mean()
    df['std'] = df['tmp'].rolling(b).std()
    df['up'] = df['ma'] + df['std']
    df['down'] = df['ma'] - df['std']
    factor = pd.Series(np.full(len(df['tmp']), np.nan))
    factor[(df['tmp']>df['upband'])] = -1
    factor[(df['tmp']<df['downband'])] = 1
    factor = factor.fillna(method = 'ffill')
    
#Bollingger Band (BBANDS) 3 动量
def factorMaker20(Data, parameter, pair):
    a = int(parameter['a'])
    b = int(parameter['b'])
    df = Data.copy()
    df['tmp'] = df['cp_adj_' + pair[0]] / df['cp_adj_' + pair[1]]
    df['ma'] = df['tmp'].rolling(a).mean()
    df['std'] = df['tmp'].rolling(b).std()
    df['up'] = df['ma'] + 2 * df['std']
    df['down'] = df['ma'] - 2 * df['std']
    factor = pd.Series(np.zeros(len(df['tmp'])))
    factor[(df['tmp']>df['up'])] = 1
    factor[(df['tmp']<df['down'])] = -1
    return factor


#Bollingger Band (BBANDS) 4 反转
def factorMaker21(Data, parameter, pair):
    a = int(parameter['a'])
    b = int(parameter['b'])
    df = Data.copy()
    df['tmp'] = df['cp_adj_' + pair[0]] / df['cp_adj_' + pair[1]]
    df['ma'] = df['tmp'].rolling(a).mean()
    df['std'] = df['tmp'].rolling(b).std()
    df['up'] = df['ma'] + 2 * df['std']
    df['down'] = df['ma'] - 2 * df['std']
    factor = pd.Series(np.full(len(df['tmp']), np.nan))
    factor[(df['tmp'] > df['upband'])] = -1
    factor[(df['tmp'] < df['downband'])] = 1
    factor = factor.fillna(method = 'ffill')
    return factor


#使用min-max归一法标准化
def normalize(col):
    return (col[-1] - col.min()) / (col.max() - col.min())

# Accumulation/Distribution (AD)
def factorMaker22(Data, parameter, pair):
    a = int(parameter['a'])
    b = int(parameter['b'])
    df = Data.copy()
    df['tmp'] = df['cp_adj_' + pair[0]] / df['cp_adj_' + pair[1]]
    df['ma'] = df['tmp'].rolling(a).mean()
    df['vol_adj_' + pair[0]] = df['volume_' + pair[0]].rolling(a).apply(normalize)
    df['vol_adj_' + pair[1]] = df['volume_' + pair[1]].rolling(a).apply(normalize)
    df['d1'] = ((df['cp_adj_' + pair[0]] - df['low_adj_' + pair[0]]) - (df['high_adj_' + pair[0]] - df['cp_adj_' + pair[0]])) / (df['high_adj_' + pair[0]] - df['low_adj_' + pair[0]]) * df['vol_adj_' + pair[0]]
    df['d2'] = ((df['cp_adj_' + pair[1]] - df['low_adj_' + pair[1]]) - (df['high_adj_' + pair[1]] - df['cp_adj_' + pair[1]])) / (df['high_adj_' + pair[1]] - df['low_adj_' + pair[1]]) * df['vol_adj_' + pair[1]]    
    df['tmp_ad'] = df['d1'] - df['d2']
    df['ma_ad'] = df['d2'].rolling(b).mean()
    factor = pd.Series(np.zeros(len(df['tmp'])))
    factor[(df['tmp'] > df['ma']) & (df['tmp_ad'] < df['ma_ad'])] = -1
    factor[(df['tmp'] < df['ma']) & (df['tmp_ad'] > df['ma_ad'])] = 1
    return factor

# Average Directional Movement (ADX)
def factorMaker23(Data, parameter, pair):    
    a = int(parameter['a'])
    b = int(parameter['b'])
    df = Data.copy()
    df['+dm1'] = df['high_adj_' + pair[0]].diff(1)
    df['+dm2'] = df['high_adj_' + pair[1]].diff(1)
    df['-dm1'] = - df['low_adj_' + pair[0]].diff(1)
    df['-dm2'] = - df['low_adj_' + pair[1]].diff(1)
    #小于零的改成0
    df.loc[df['+dm1'] <= 0, ["+dm1"]] = 0
    df.loc[df['+dm2'] <= 0, ["+dm2"]] = 0
    df.loc[df['-dm1'] <= 0, ["-dm1"]] = 0
    df.loc[df['-dm2'] <= 0, ["-dm2"]] = 0
    #相对较小的一个也改成0
    df.loc[df['+dm1'] <= df['-dm1'], ['+dm1']] = 0
    df.loc[df['+dm1'] >= df['-dm1'], ['-dm1']] = 0
    df.loc[df['+dm2'] <= df['-dm2'], ['+dm2']] = 0
    df.loc[df['+dm2'] >= df['-dm2'], ['-dm2']] = 0
    df['+adm1'] = df['+dm1'].ewm(halflife = a).mean()
    df['+adm2'] = df['+dm2'].ewm(halflife = a).mean()
    df['-adm1'] = df['-dm1'].ewm(halflife = a).mean()
    df['-adm2'] = df['+dm2'].ewm(halflife = a).mean()
    df['hml1'] = df['high_adj_' + pair[0]] - df['low_adj_' + pair[0]]
    df['hmc1'] = df['high_adj_' + pair[0]] - df['cp_adj_' + pair[0]].shift(1)
    df['cml1'] = df['cp_adj_' + pair[0]].shift(1) - df['low_adj_' + pair[0]]
    df['hml2'] = df['high_adj_' + pair[1]] - df['low_adj_' + pair[1]]
    df['hmc2'] = df['high_adj_' + pair[1]] - df['cp_adj_' + pair[1]].shift(1)
    df['cml2'] = df['cp_adj_' + pair[1]].shift(a) - df['low_adj_' + pair[1]]
    df['tr1'] = df[['hml1','hmc1','cml1']].max(axis = 1)
    df['tr2'] = df[['hml2','hmc2','cml2']].max(axis = 1)
    df['atr1'] = df['tr1'].ewm(halflife = a).mean()
    df['atr2'] = df['tr2'].ewm(halflife = a).mean()
    df['+di1'] = df['+adm1'] / df['atr1'] * 100
    df['+di2'] = df['+adm2'] / df['atr2'] * 100
    df['-di1'] = df['-adm1'] / df['atr1'] * 100
    df['-di2'] = df['-adm2'] / df['atr2'] * 100
    df['dx1'] = abs(df['+di1'] - df['-di1']) / (df['+di1'] + df['-di1']) * 100
    df['dx2'] = abs(df['+di2'] - df['-di2']) / (df['+di2'] + df['-di2']) * 100
    df['adx1'] = df['dx1'].ewm(halflife = b).mean()
    df['adx2'] = df['dx2'].ewm(halflife = b).mean()
    return 0

# Average True Range (ATR) 1
def factorMaker24(Data, parameter, pair):    
    a = int(parameter['a'])
    b = int(parameter['b'])
    df = Data.copy()
    df['high'] = df['high_adj_' + pair[0]] / df['low_adj_' + pair[1]]
    df['low'] = df['low_adj_' + pair[0]] / df['high_adj_' + pair[1]]
    df['tmp'] = df['cp_adj_' + pair[0]] / df['cp_adj_' + pair[1]]
    df['ma'] = df['tmp'].rolling(a).mean()
    df['hml'] = df['high'] - df['low']
    df['hmc'] = df['high'] - df['tmp'].shift(1)
    df['cml'] = df['tmp'].shift(1) - df['low']
    df['tr'] = df[['hml','hmc','cml']].max(axis = 1)
    df['atr'] = df['tr'].ewm(halflife = b).mean()
    df['high_line'] = df['ma'] + df['atr']
    df['down_line'] = df['ma'] - df['atr']
    df['mid_up_line'] = df['ma'] + 0.5 * df['atr']
    df['mid_down_line'] = df['ma'] - 0.5 * df['atr']
    factor = pd.Series(np.full(len(df['tmp']), np.nan))
    factor[(df['tmp'] > df['mid_down_line']) & (df['tmp'] < df['mid_up_line'])] = 0
    factor[(df['tmp'] < df['down_line'])] = 1
    factor[(df['tmp'] > df['high_line'])] = -1
    factor = factor.fillna(method = 'ffill')
    return factor

# Average True Range (ATR) 1
def factorMaker24(Data, parameter, pair):    
    a = int(parameter['a'])
    b = int(parameter['b'])
    df = Data.copy()
    df['high'] = df['high_adj_' + pair[0]] / df['low_adj_' + pair[1]]
    df['low'] = df['low_adj_' + pair[0]] / df['high_adj_' + pair[1]]
    df['tmp'] = df['cp_adj_' + pair[0]] / df['cp_adj_' + pair[1]]
    df['ma'] = df['tmp'].rolling(a).mean()
    df['hml'] = df['high'] - df['low']
    df['hmc'] = df['high'] - df['tmp'].shift(1)
    df['cml'] = df['tmp'].shift(1) - df['low']
    df['tr'] = df[['hml','hmc','cml']].max(axis = 1)
    df['atr'] = df['tr'].ewm(halflife = b).mean()
    df['high_line'] = df['ma'] + df['atr']
    df['down_line'] = df['ma'] - df['atr']
    df['mid_up_line'] = df['ma'] + 0.5 * df['atr']
    df['mid_down_line'] = df['ma'] - 0.5 * df['atr']
    factor = pd.Series(np.full(len(df['tmp']), np.nan))
    factor[(df['tmp'] > df['mid_down_line']) & (df['tmp'] < df['mid_up_line'])] = 0
    factor[(df['tmp'] < df['down_line'])] = -1
    factor[(df['tmp'] > df['high_line'])] = 1
    factor = factor.fillna(method = 'ffill')
    return factor

# Commodity Channel Index (CCI)
def factorMaker25(Data, parameter, pair):  
    a = int(parameter['a'])
    b = int(parameter['b'])
    df = Data.copy()
    df['m1'] = (df['high_adj_' + pair[0]] + df['low_adj_' + pair[0]] + df['cp_adj_' + pair[0]]) / 3
    df['m2'] = (df['high_adj_' + pair[1]] + df['low_adj_' + pair[1]] + df['cp_adj_' + pair[1]]) / 3
    df['a1'] = df['m1'].rolling(a).mean()
    df['a2'] = df['m2'].rolling(a).mean()
    df['m1-a1'] = abs(df['m1'] - df['a1'])
    df['m2-a2'] = abs(df['m2'] - df['a2'])
    df['d1'] = df['m1-a1'].rolling(b).mean()
    df['d2'] = df['m2-a2'].rolling(b).mean()
    df['cci1'] = (df['m1'] - df['a1']) / (0.015 * df['d1'])
    df['cci2'] = (df['m2'] - df['a2']) / (0.015 * df['d2'])
    df['delta_cci'] = df['cci1'] - df['cci2']
    factor = pd.Series(np.zeros(len(df['cci'])))
    factor[(df['delta_cci'] > 150)] = 1
    factor[(df['delta_cci'] < -150)] = -1
    return factor

# Chande Momentum Oscillator (CMO)
def factorMaker26(Data, parameter, pair):  
    a = int(parameter['a'])
    b = int(parameter['b'])
    df = Data.copy()
    df['delta1'] = df['cp_adj_' + pair[0]].diff(1)
    df['+delta1'] = df[df['delta1'] > 0]['delta1']
    df['-delta1'] = - df[df['delta1'] < 0]['delta1']
    df['+delta1'] = df['+delta1'].fillna(0)
    df['-delta1'] = df['-delta1'].fillna(0)
    df['possum1'] = df['+delta1'].rolling(a).sum()
    df['negsum1'] = df['-delta1'].rolling(a).sum()
    df['cmo1'] = (df['possum1'] - df['negsum1']) /  (df['possum1'] + df['negsum1']) * 100
    df['delta2'] = df['cp_adj_' + pair[1]].diff(1)
    df['+delta2'] = df[df['delta2'] > 0]['delta2']
    df['-delta2'] = - df[df['delta2'] < 0]['delta2']
    df['+delta2'] = df['+delta2'].fillna(0)
    df['-delta2'] = df['-delta2'].fillna(0)
    df['possum2'] = df['+delta2'].rolling(a).sum()
    df['negsum2'] = df['-delta2'].rolling(a).sum()
    df['cmo2'] = (df['possum2'] - df['negsum2']) /  (df['possum2'] + df['negsum2']) * 100
    df['delta_cmo'] = df['cmo1'] - df['cmo2']
    factor = pd.Series(np.zeros(len(df['cci'])))
    factor[(df['delta_cmo'] > 75)] = -1
    factor[(df['delta_cci'] < -75)] = 1
    return factor

# Double Exponential Moving Average (DEMA)
def factorMaker27(Data, parameter, pair):  
    a = int(parameter['a'])
    b = int(parameter['b'])
    df = Data.copy()
    df['tmp'] = df['cp_adj_' + pair[0]] / df['cp_adj_' + pair[1]]
    df['ema'] = df['tmp'].ewm(halflife = a).mean()
    df['emaema'] = df['ema'].ewm(halflife = a).mean()
    df['dema'] = 2 * df['ema'] - df['emaema']
    factor = pd.Series(np.zeros(len(df['tmp'])))
    factor[(df['tmp'] > df['dema'])] = 1
    factor[(df['tmp'] < df['dema'])] = -1
    return factor

# Ichimoku (ICH)
def factorMaker28(Data, parameter, pair):  
    a = int(parameter['a'])
    b = int(parameter['b'])
    df = Data.copy()
    df['high'] = df['high_adj_' + pair[0]] / df['low_adj_' + pair[1]]
    df['low'] = df['low_adj_' + pair[0]] / df['high_adj_' + pair[1]]
    df['tmp'] = df['cp_adj_' + pair[0]] / df['cp_adj_' + pair[1]]
    df['highest1'] = df['high'].rolling(a).max()
    df['highest2'] = df['high'].rolling(b).max()
    df['lowest1'] = df['low'].rolling(a).min()
    df['lowest2'] = df['low'].rolling(b).min()
    df['tline'] = (df['highest1'] + df['lowest1']) / 2
    df['sline'] = (df['highest2'] + df['lowest2']) / 2
    df['ls1'] = (df['tline'] + df['sline']) / 2
    df['ls2'] = (df['highest2'].shift(b) + df['lowest2'].shift(b)) / 2
    factor = pd.Series(np.zeros(len(df['tmp'])))
    factor[(df['tmp'] > df['ls1']) & (df['tmp'] > df['ls2']) & (df['tline'] > df['sline'])] = 1
    factor[(df['tmp'] < df['ls1']) & (df['tmp'] < df['ls2']) & (df['tline'] < df['sline'])] = -1
    return factor

# Linear Regression (LR)
import statsmodels.api as sm
def ln_regression1(col):
    X = np.arange(1,a + 1,1)
    y = col
    est = sm.OLS(y, sm.add_constant(X)).fit()
    return est.predict((1,a + 1))[0]

def factorMaker29(Data, parameter, pair):
    a = int(parameter['a'])
    b = int(parameter['b'])
    df = Data.copy()
    df['tmp'] = df['cp_adj_' + pair[0]] / df['cp_adj_' + pair[1]]
    df['tmp'] = df['tmp'].rolling(b).mean()
    df['pred'] = df['tmp'].rolling(a).apply(ln_regression1)
    factor = pd.Series(np.zeros(len(df['tmp'])))
    factor[(df['tmp'] > df['pred'])] = 1
    factor[(df['tmp'] < df['pred'])] = -1
    return factor    

def factorMaker30(Data, parameter, pair):
    a = int(parameter['a'])
    b = int(parameter['b'])
    df = Data.copy()
    df['tmp'] = df['cp_adj_' + pair[0]] / df['cp_adj_' + pair[1]]
    df['tmp'] = df['tmp'].rolling(b).mean()
    df['pred'] = df['tmp'].rolling(a).apply(ln_regression1)
    factor = pd.Series(np.zeros(len(df['tmp'])))
    factor[(df['tmp'] > df['pred'])] = -1
    factor[(df['tmp'] < df['pred'])] = 1
    return factor    
    
# Linear Regression Angle (LRA) & SLOPE
def ln_regression2(col):
    X = np.arange(1,a + 1,1)
    y = col
    est = sm.OLS(y, sm.add_constant(X)).fit()
    return est.params[1]
    
def factorMaker31(Data, parameter, pair):
    a = int(parameter['a'])
    b = int(parameter['b'])
    df = Data.copy()
    df['tmp'] = df['cp_adj_' + pair[0]] / df['cp_adj_' + pair[1]]
    df['slope'] = df['tmp'].rolling(a).apply(ln_regression2)
    factor = pd.Series(np.zeros(len(df['tmp'])))
    factor[(df['slope'] > b)] = -1
    factor[(df['tmp'] < -b)] = 1
    return factor   

def factorMaker32(Data, parameter, pair):
    a = int(parameter['a'])
    b = int(parameter['b'])
    df = Data.copy()
    df['tmp'] = df['cp_adj_' + pair[0]] / df['cp_adj_' + pair[1]]
    df['slope'] = df['tmp'].rolling(a).apply(ln_regression2)
    factor = pd.Series(np.zeros(len(df['tmp'])))
    factor[(df['slope'] > b)] = 1
    factor[(df['tmp'] < -b)] = -1
    return factor   

# Max (MAX) Min(MIN) Price Channel PC
def factorMaker33(Data, parameter, pair):
    a = int(parameter['a'])
    b = int(parameter['b'])
    df = Data.copy()
    df['high'] = df['high_adj_' + pair[0]] / df['low_adj_' + pair[1]]
    df['low'] = df['low_adj_' + pair[0]] / df['high_adj_' + pair[1]]
    df['tmp'] = df['cp_adj_' + pair[0]] / df['cp_adj_' + pair[1]]
    df['highest'] = df['high'].rolling(a).max()
    df['lowest'] = df['low'].rolling(a).min()
    factor = pd.Series(np.zeros(len(df['tmp'])))
    factor[(df['tmp'] > df['highest'])] = 1
    factor[(df['tmp'] < df['lowest'])] = -1
    return factor

# Money  Flow Index (MFI)
def factorMaker34(Data, parameter, pair):
    a = int(parameter['a'])
    b = int(parameter['b'])
    df = Data.copy()
    df['pivot1'] = (df['high_adj_' + pair[0]] + df['cp_adj_' + pair[0]] + df['low_adj_' + pair[0]]) / 3
    df['pivot2'] = (df['high_adj_' + pair[1]] + df['cp_adj_' + pair[1]] + df['low_adj_' + pair[1]]) / 3
    df['sign1'] = np.sign(df['cp_adj_' + pair[0]].diff(1))
    df['sign2'] = np.sign(df['cp_adj_' + pair[1]].diff(1))
    df['mf1'] = df['sign1'] * df['pivot1'] * df['volume_' + pair[0]]
    df['mf2'] = df['sign2'] * df['pivot2'] * df['volume_' + pair[1]]
    df['+mf1'] = df[df['mf1'] > 0]['mf1']
    df['-mf1'] = - df[df['mf1'] < 0]['mf1']
    df['+mf1'] = df['+mf1'].fillna(0)
    df['-mf1'] = df['-mf1'].fillna(0)
    df['+mf2'] = df[df['mf2'] > 0]['mf2']
    df['-mf2'] = - df[df['mf2'] < 0]['mf2']
    df['+mf2'] = df['+mf2'].fillna(0)
    df['-mf2'] = df['-mf2'].fillna(0)
    df['spf1'] = df['+mf1'].rolling(a).sum()
    df['snf1'] = df['-mf1'].rolling(a).sum()
    df['spf2'] = df['+mf2'].rolling(a).sum()
    df['snf2'] = df['-mf2'].rolling(a).sum()
    df['mfi1'] = 100 - (100 / (1 + df['spf1'] / df['snf1']))
    df['mfi2'] = 100 - (100 / (1 + df['spf2'] / df['snf2']))
    df['ratio'] = df['mfi1'] / df['mfi2']
    factor = pd.Series(np.zeros(len(df['ratio'])))
    factor[(df['ratio'] > b)] = -1
    factor[(df['ratio'] < b)] = 1
    return factor

# Midpoint (MIDPNT) 1
def factorMaker35(Data, parameter, pair):
    a = int(parameter['a'])
    b = int(parameter['b'])
    df = Data.copy()
    df['high'] = df['high_adj_' + pair[0]] / df['low_adj_' + pair[1]]
    df['low'] = df['low_adj_' + pair[0]] / df['high_adj_' + pair[1]]
    df['tmp'] = df['cp_adj_' + pair[0]] / df['cp_adj_' + pair[1]]
    df['highest'] = df['high'].rolling(a).max()
    df['lowest'] = df['low'].rolling(a).min()
    df['midpoint'] = (df['highest'] + df['lowest']) / 2
    factor = pd.Series(np.zeros(len(df['tmp'])))
    factor[(df['tmp'] > df['midpoint'])] = -1
    factor[(df['tmp'] < df['midpoint'])] = 1
    return factor

# Midpoint (MIDPNT) 2
def factorMaker36(Data, parameter, pair):
    a = int(parameter['a'])
    b = int(parameter['b'])
    df = Data.copy()
    df['high'] = df['high_adj_' + pair[0]] / df['low_adj_' + pair[1]]
    df['low'] = df['low_adj_' + pair[0]] / df['high_adj_' + pair[1]]
    df['tmp'] = df['cp_adj_' + pair[0]] / df['cp_adj_' + pair[1]]
    df['highest'] = df['high'].rolling(a).max()
    df['lowest'] = df['low'].rolling(a).min()
    df['midpoint'] = (df['highest'] + df['lowest']) / 2
    factor = pd.Series(np.zeros(len(df['tmp'])))
    factor[(df['tmp'] > df['midpoint'])] = 1
    factor[(df['tmp'] < df['midpoint'])] = -1
    return factor

# Midprice (MIDPRI)
def factorMaker37(Data, parameter, pair):
    a = int(parameter['a'])
    b = int(parameter['b'])
    df = Data.copy()
    df['high'] = df['high_adj_' + pair[0]] / df['low_adj_' + pair[1]]
    df['low'] = df['low_adj_' + pair[0]] / df['high_adj_' + pair[1]]
    df['tmp'] = df['cp_adj_' + pair[0]] / df['cp_adj_' + pair[1]]
    df['midprice'] = (df['high'] + df['low']) / 2
    df['ma'] = df['midprice'].rolling(a).mean()
    factor = pd.Series(np.zeros(len(df['tmp'])))
    factor[(df['tmp'] > df['ma'])] = 1
    factor[(df['tmp'] < df['ma'])] = -1
    return factor

# Midprice (MIDPRI) 2
def factorMaker38(Data, parameter, pair):
    a = int(parameter['a'])
    b = int(parameter['b'])
    df = Data.copy()
    df['high'] = df['high_adj_' + pair[0]] / df['low_adj_' + pair[1]]
    df['low'] = df['low_adj_' + pair[0]] / df['high_adj_' + pair[1]]
    df['tmp'] = df['cp_adj_' + pair[0]] / df['cp_adj_' + pair[1]]
    df['midprice'] = (df['high'] + df['low']) / 2
    df['ma'] = df['midprice'].rolling(a).mean()
    factor = pd.Series(np.zeros(len(df['tmp'])))
    factor[(df['tmp'] > df['ma'])] = -1
    factor[(df['tmp'] < df['ma'])] = 1
    return factor

# Momentum (MOM) / Rate of Change
def factorMaker39(Data, parameter, pair):
    a = int(parameter['a'])
    b = int(parameter['b'])
    df = Data.copy()
    df['tmp1'] = df['cp_adj_' + pair[0]].pct_change(a)
    df['tmp2'] = df['cp_adj_' + pair[1]].pct_change(a)
    factor = df['tmp1'] - df['tmp2']
    factor = smoothfactor(factor, cut, b)
    factor = np.sign(factor)
    return factor

#Adaptive Moving Average (AMA)
def cal_ama2(price):
    global i
    global ama
    ama = df.at[i,'coef'] * (price - ama) + ama 
    i += 1
    return ama

def factorMaker40(Data, parameter, pair):
    a = int(parameter['a'])
    b = int(parameter['b'])
    df = Data.copy()
    df['tmp'] = df['cp_adj_' + pair[0]] / df['cp_adj_' + pair[1]]
    df['direc'] = df['tmp'].diff(1)
    df['absdirec'] = abs(df['direc'])
    df['sumdirec'] = df['direc'].rolling(a).sum()
    df['sumabsdirec'] = df['absdirec'].rolling(a).sum()
    df['er'] = abs(df['sumdirec'] / df['sumabsdirec'])
    fc = 2 / (a + 1)
    sc = 2 / (b + 1)
    df['coef'] = (df['er'] * (fc - sc) + sc)
    df['coef'] = df['coef'].fillna(method = 'bfill')
    i = 0
    ama = df.at[0,'tmp']
    df['ama'] = df['tmp'].apply(cal_ama2)
    factor = pd.Series(np.zeros(len(df['tmp'])))
    factor[(df['tmp'] > df['ama'])] = 1
    factor[(df['tmp'] < df['ama'])] = -1
    return factor    

# RSI相对强弱指标
def factorMaker41(Data, parameter, pair):
    a = int(parameter['a'])
    b = int(parameter['b'])
    df = Data.copy()
    df['tmp1'] = df['cp_adj_' + pair[0]].pct_change()
    df['tmp2'] = df['cp_adj_' + pair[1]].pct_change()
    df['tmp1abs'] = abs(df['tmp1'])
    df['tmp2abs'] = abs(df['tmp2'])
    df['rise_sum1'] =  (df['tmp1'].rolling(a).sum() + df['tmp1abs'].rolling(a).sum()) / 2
    df['rise_sum2'] =  (df['tmp2'].rolling(a).sum() + df['tmp2abs'].rolling(a).sum()) / 2
    df['rsi1'] = df['rise_sum1'] / df['tmp1abs'].rolling(a).sum()
    df['rsi2'] = df['rise_sum2'] / df['tmp2abs'].rolling(a).sum()
    factor = df['rsi1'] - df['rsi2']
    factor = smoothfactor(factor, cut, b)
    factor = np.sign(factor)
    return factor

# Exponential (EMA)
def factorMaker42(Data, parameter, pair):
    a = int(parameter['a'])
    b = int(parameter['b'])
    df = Data.copy()
    df['tmp'] = df['cp_adj_' + pair[0]] / df['cp_adj_' + pair[1]]
    df['ema'] = df['tmp'].ewm(span = a).mean()
    factor = pd.Series(np.zeros(len(df['tmp'])))
    factor[(df['tmp'] > df['ema'])] = 1
    factor[(df['tmp'] < df['ema'])] = -1
    return factor

# Moving Average Convergence Divergence (MACD)
def factorMaker43(Data, parameter, pair):
    a = int(parameter['a'])
    b = int(parameter['b'])
    df = Data.copy()
    df['tmp'] = df['cp_adj_' + pair[0]] / df['cp_adj_' + pair[1]]
    df['fma'] = df['tmp'].ewm(span = a).mean()
    df['sma'] = df['tmp'].ewm(span = b).mean()
    df['dif'] = df['fma'] - df['sma']
    df['dea'] = df['tmp'].ewm(span = a).mean()
    df['macd'] = df['dif'] - df['dea']
    factor = pd.Series(np.zeros(len(df['tmp'])))
    factor[(df['dif'] > 0) & (df['dea'] > 0)  & (df['macd'] > 0)] = 1
    factor[(df['dif'] < 0) & (df['dea'] < 0)  & (df['macd'] < 0)] = -1
    return factor

# Simple Moving Average (SMA)
def factorMaker44(Data, parameter, pair):
    a = int(parameter['a'])
    b = int(parameter['b'])
    df = Data.copy()
    df['tmp'] = df['cp_adj_' + pair[0]] / df['cp_adj_' + pair[1]]
    df['sma'] = df['tmp'].rolling(a).mean()
    factor = pd.Series(np.zeros(len(df['tmp'])))
    factor[(df['tmp'] > df['sma'])] = 1
    factor[(df['tmp'] < df['sma'])] = -1
    return factor

# T3 (T3)    
def gd(tmp, a, b):
    ema1 = tmp.ewm(span = a).mean()
    ema2 = ema1.ewm(span = a).mean()
    return ema1 * (1 + b) - ema2 * b

def factorMaker45(Data, parameter, pair):
    a = int(parameter['a'])
    b = int(parameter['b'])
    df = Data.copy()
    df['tmp'] = df['cp_adj_' + pair[0]] / df['cp_adj_' + pair[1]]
    df['t3'] = gd(gd(gd(df['tmp'], a, b), a, b), a, b)
    factor = pd.Series(np.zeros(len(df['tmp'])))
    factor[(df['tmp'] > df['t3'])] = 1
    factor[(df['tmp'] < df['t3'])] = -1
    return factor

# Triple Exponential Moving Average (TEMA)
def factorMaker46(Data, parameter, pair):
    a = int(parameter['a'])
    b = int(parameter['b'])
    df = Data.copy()
    df['tmp'] = df['cp_adj_' + pair[0]] / df['cp_adj_' + pair[1]]
    df['ema1'] = df['tmp'].ewm(span = a).mean()
    df['ema2'] = df['ema1'].ewm(span = a).mean()
    df['ema3'] = df['ema2'].ewm(span = a).mean()
    df['tema'] = 3 * df['ema1'] - 3 * df['ema2'] + df['ema3']
    factor = pd.Series(np.zeros(len(df['tmp'])))
    factor[(df['tmp'] > df['tema'])] = 1
    factor[(df['tmp'] < df['tema'])] = -1
    return factor

# Triangular Moving Average (TRIMA)
def factorMaker47(Data, parameter, pair):
    a = int(parameter['a'])
    b = int(parameter['b'])
    df = Data.copy()
    df['tmp'] = df['cp_adj_' + pair[0]] / df['cp_adj_' + pair[1]]
    df['smam'] = df['tmp'].rolling(a).mean()
    df['ma'] = df['sma'].rolling(b).mean()
    factor = pd.Series(np.zeros(len(df['tmp'])))
    factor[(df['tmp'] > df['ma'])] = 1
    factor[(df['tmp'] < df['ma'])] = -1
    return factor
    
# Triple Exponential Moving Average Oscillator (TRIX)
def factorMaker48(Data, parameter, pair):
    a = int(parameter['a'])
    b = int(parameter['b'])
    df = Data.copy()
    df['tmp'] = df['cp_adj_' + pair[0]] / df['cp_adj_' + pair[1]]
    df['ema1'] = df['tmp'].ewm(span = a).mean()
    df['ema2'] = df['ema1'].ewm(span = a).mean()
    df['ema3'] = df['ema2'].ewm(span = a).mean()
    df['trix'] = df['ema3'].pct_change(1)
    df['trma'] = df['trix'].rolling(b).mean()
    factor = pd.Series(np.zeros(len(df['tmp'])))
    factor[(df['trix'] > df['trma'])] = 1
    factor[(df['trix'] < df['trma'])] = -1
    return factor    

# Weighted Moving Average (WMA)
def wma(col, a):
    x = np.arange(a, 0, -1) / (a * (a + 1) / 2)
    return np.dot(col, x)

def factorMaker49(Data, parameter, pair):
    a = int(parameter['a'])
    b = int(parameter['b'])
    df = Data.copy()
    df['tmp'] = df['cp_adj_' + pair[0]] / df['cp_adj_' + pair[1]]
    x = np.arange(a, 0, -1) / (a * (a + 1) / 2)
    df['wma'] = df['tmp'].rolling(a).apply(lambda col: np.dot(col, x))
    factor = pd.Series(np.zeros(len(df['tmp'])))
    factor[(df['tmp'] > df['wma'])] = 1
    factor[(df['tmp'] < df['wma'])] = -1
    return factor    

# Normalized Average True Range (NATR) 1
def factorMaker50(Data, parameter, pair):    
    a = int(parameter['a'])
    b = int(parameter['b'])
    df = Data.copy()
    df['high'] = df['high_adj_' + pair[0]] / df['low_adj_' + pair[1]]
    df['low'] = df['low_adj_' + pair[0]] / df['high_adj_' + pair[1]]
    df['tmp'] = df['cp_adj_' + pair[0]] / df['cp_adj_' + pair[1]]
    df['ma'] = df['tmp'].rolling(a).mean()
    df['hml'] = df['high'] - df['low']
    df['hmc'] = df['high'] - df['tmp'].shift(1)
    df['cml'] = df['tmp'].shift(1) - df['low']
    df['tr'] = df[['hml','hmc','cml']].max(axis = 1)
    df['atr'] = df['tr'].ewm(halflife = b).mean()
    df['natr'] = df['atr'] / df['tmp']
    return 0

def factorMaker51(Data, parameter, pair):    
    a = int(parameter['a'])
    b = int(parameter['b'])
    df = Data.copy()
    df['hml1'] = df['high_adj_' + pair[0]] - df['low_adj_' + pair[0]]
    df['hmc1'] = df['high_adj_' + pair[0]] - df['cp_adj_' + pair[0]].shift(1)
    df['cml1'] = df['cp_adj_' + pair[0]].shift(1) - df['low_adj_' + pair[0]]
    df['hml2'] = df['high_adj_' + pair[1]] - df['low_adj_' + pair[1]]
    df['hmc2'] = df['high_adj_' + pair[1]] - df['cp_adj_' + pair[1]].shift(1)
    df['cml2'] = df['cp_adj_' + pair[1]].shift(1) - df['low_adj_' + pair[1]]
    df['tr1'] = df[['hml1','hmc1','cml1']].max(axis = 1)
    df['tr2'] = df[['hml2','hmc2','cml2']].max(axis = 1)
    df['atr1'] = df['tr1'].ewm(halflife = b).mean()
    df['atr2'] = df['tr2'].ewm(halflife = b).mean()
    df['natr1'] = df['atr'] / df['tmp']
    df['natr2'] = df['atr'] / df['tmp']
    factor = df['natr1'] - df['natr2']
    factor = smoothfactor(factor, cut, b)
    factor = np.sign(factor)
    return factor

# On Balance Volume (OBV)
def factorMaker52(Data, parameter, pair):    
    a = int(parameter['a'])
    b = int(parameter['b'])
    df = Data.copy()
    df['sign1'] = np.sign(df['cp_adj_' + pair[0]].pct_change(1))
    df['vol1'] = df['volume_'+ pair[0]] * df['sign1']
    df['obv1'] = df['vol1'].cumsum()
    df['sign2'] = np.sign(df['cp_adj_' + pair[1]].pct_change(1))
    df['vol2'] = df['volume_'+ pair[1]] * df['sign2']
    df['obv2'] = df['vol2'].cumsum()
    df['tmp'] = df['obv1'] / df['obv2']
    df['tmpma'] = df['tmp'].rolling(a).mean()
    factor = pd.Series(np.zeros(len(df['tmp'])))
    factor[(df['tmp'] > df['tmpma'])] = 1
    factor[(df['tmp'] < df['tmpma'])] = -1
    return factor    

# Percent Price Oscillator (PPO)
def factorMaker53(Data, parameter, pair):
    a = int(parameter['a'])
    b = int(parameter['b'])
    df = Data.copy()
    df['fma1'] = df['cp_adj_' + pair[0]].ewm(span = a).mean()
    df['sma1'] = df['cp_adj_' + pair[0]].ewm(span = b).mean()
    df['fma2'] = df['cp_adj_' + pair[1]].ewm(span = a).mean()
    df['sma2'] = df['cp_adj_' + pair[1]].ewm(span = b).mean()
    df['dif1'] = df['fma1'] - df['sma1']
    df['dif2'] = df['fma2'] - df['sma2']
    df['ppo1'] = df['dif1'] / df['sma1']
    df['ppo2'] = df['dif2'] / df['sma2']
    df['delta_ppo'] = df['ppo1'] - df['ppo2']
    factor = pd.Series(np.zeros(len(df['delta_ppo'])))
    factor[(df['delta_ppo'] > 0)] = 1
    factor[(df['delta_ppo'] < 0)] = -1
    return factor

# Stochastic (STOCH)
def factorMaker54(Data, parameter, pair):
    a = int(parameter['a'])
    b = int(parameter['b'])
    df = Data.copy()
    df['high'] = df['high_adj_' + pair[0]] / df['low_adj_' + pair[1]]
    df['low'] = df['low_adj_' + pair[0]] / df['high_adj_' + pair[1]]
    df['tmp'] = df['cp_adj_' + pair[0]] / df['cp_adj_' + pair[1]]
    df['highest'] = df['high'].rolling(a).max()
    df['lowest'] = df['low'].rolling(a).min()
    df['K'] =  (df['tmp'] - df['lowest']) / (df['highest'] - df['lowest'])
    df['fk'] = df['K'].rolling(a).mean()
    df['sk'] = df['fk'].rolling(a).mean()
    df['sd'] = df['sk'].rolling(b).mean()
    factor = pd.Series(np.zeros(len(df['sk'])))
    factor[(df['sk'] > df['sd'])] = 1
    factor[(df['sk'] < df['sd'])] = -1
    return factor

# Stochastic Fast (StochF)
def factorMaker55(Data, parameter, pair):
    a = int(parameter['a'])
    b = int(parameter['b'])
    df = Data.copy()
    df['high'] = df['high_adj_' + pair[0]] / df['low_adj_' + pair[1]]
    df['low'] = df['low_adj_' + pair[0]] / df['high_adj_' + pair[1]]
    df['tmp'] = df['cp_adj_' + pair[0]] / df['cp_adj_' + pair[1]]
    df['highest'] = df['high'].rolling(a).max()
    df['lowest'] = df['low'].rolling(a).min()
    df['K'] =  (df['tmp'] - df['lowest']) / (df['highest'] - df['lowest'])
    df['fk'] = df['K'].rolling(a).mean()
    df['fd'] = df['fk'].rolling(b).mean()
    factor = pd.Series(np.zeros(len(df['fk'])))
    factor[(df['fk'] > df['fd'])] = 1
    factor[(df['fk'] < df['fd'])] = -1
    return factor

# Ultimate Oscillator (ULTOSC)
def factorMaker56(Data, parameter, pair):
    a = int(parameter['a'])
    b = int(parameter['b'])
    df = Data.copy()
    df['cp1'] = df['cp_adj_' + pair[0]].shift(1)
    df['cp2'] = df['cp_adj_' + pair[1]].shift(1)
    df['min1'] = df[['cp1','low_adj_' + pair[0]]].min(axis = 1)
    df['min2'] = df[['cp2','low_adj_' + pair[1]]].min(axis = 1)
    df['bp1'] = df['cp_adj_' + pair[0]] - df['min1']
    df['bp2'] = df['cp_adj_' + pair[1]] - df['min2']
    df['hml1'] = df['high_adj_' + pair[0]] - df['low_adj_' + pair[0]]
    df['hmc1'] = df['high_adj_' + pair[0]] - df['cp_adj_' + pair[0]].shift(1)
    df['cml1'] = df['cp_adj_' + pair[0]].shift(1) - df['low_adj_' + pair[0]]
    df['hml2'] = df['high_adj_' + pair[1]] - df['low_adj_' + pair[1]]
    df['hmc2'] = df['high_adj_' + pair[1]] - df['cp_adj_' + pair[1]].shift(1)
    df['cml2'] = df['cp_adj_' + pair[1]].shift(1) - df['low_adj_' + pair[1]]
    df['tr1'] = df[['hml1','hmc1','cml1']].max(axis = 1)
    df['tr2'] = df[['hml2','hmc2','cml2']].max(axis = 1)
    df['a11'] = df['bp1'].rolling(a).sum() / df['tr1'].rolling(a).sum()
    df['a21'] = df['bp1'].rolling(2*a).sum() / df['tr1'].rolling(2*a).sum()
    df['a31'] = df['bp1'].rolling(4*a).sum() / df['tr1'].rolling(4*a).sum()
    df['a12'] = df['bp2'].rolling(a).sum() / df['tr2'].rolling(a).sum()
    df['a22'] = df['bp2'].rolling(2*a).sum() / df['tr2'].rolling(2*a).sum()
    df['a32'] = df['bp2'].rolling(4*a).sum() / df['tr2'].rolling(4*a).sum()
    df['ultosc1'] = 100 * (4 * df['a11'] + 2 * df['a21'] + df['a31']) / 7
    df['ultosc2'] = 100 * (4 * df['a12'] + 2 * df['a22'] + df['a32']) / 7
    factor = df['ultosc1'] - df['ultosc2']
    factor = smoothfactor(factor, cut, b)
    factor = np.sign(factor)
    return factor

# Volume Weighted Average Price (VWAP)
def factorMaker57(Data, parameter, pair):
    a = int(parameter['a'])
    b = int(parameter['b'])
    df = Data.copy()
    df['qsum1'] = df['volume_' + pair[0]].rolling(a).sum()
    df['qp1'] = df['volume_' + pair[0]] * df['cp_adj_' + pair[0]]
    df['qpsum1'] = df['qp1'].rolling(a).sum()
    df['vwap1'] = df['qpsum1'] / df['qsum1']
    df['qsum2'] = df['volume_' + pair[1]].rolling(a).sum()
    df['qp2'] = df['volume_' + pair[1]] * df['cp_adj_' + pair[1]]
    df['qpsum2'] = df['qp2'].rolling(a).sum()
    df['vwap2'] = df['qpsum2'] / df['qsum2']
    df['tmp'] = df['vwap1'] / df['vwap2']
    df['ma'] = df['tmp'].rolling(b).mean()
    factor = pd.Series(np.zeros(len(df['tmp'])))
    factor[(df['tmp'] > df['ma'])] = 1
    factor[(df['tmp'] < df['ma'])] = -1
    return factor

# Williams % R (WillR)
def factorMaker58(Data, parameter, pair):
    a = int(parameter['a'])
    b = int(parameter['b'])
    df = Data.copy()
    df['highest1'] = df['high_adj_' + pair[0]].rolling(a).max()
    df['lowest1'] = df['low_adj_' + pair[0]].rolling(a).min()
    df['highest2'] = df['high_adj_' + pair[1]].rolling(a).max()
    df['lowest2'] = df['low_adj_' + pair[1]].rolling(a).min()
    df['%r1'] = -100 * (df['highest1'] - df['cp_adj_' + pair[0]]) / (df['highest1'] - df['lowest1'])
    df['%r2'] = -100 * (df['highest2'] - df['cp_adj_' + pair[1]]) / (df['highest2'] - df['lowest2'])
    factor = df['%r1'] - df['%r2']
    factor = smoothfactor(factor, cut, b)
    factor = np.sign(factor)
    return factor 

# Parabolic Sar (SAR) 
def cal_sar(col):    
    global trend
    global sar
    global af
    global ep
    if trend == 1:
        # 涨势
        sar = sar + af * (ep - sar)
        if sar > col.min():
            trend = 4
        else:
            ep = col[a - b:].max()
            if col[-1] == ep:
                af += 0.02
            if af >= 0.2:
                af = 0.2
    elif trend == 2: 
        # 跌势
        sar = sar + af * (ep - sar)
        if sar < col.max():
            trend = 3
        else:
            ep = col[a - b:].min()
            if col[-1] == ep:
                af += 0.02
            if af >= 0.2:
                af = 0.2
    elif trend == 3:
        # 首次涨势
        sar = col[a - b:].min()
        if sar > col.min():
            trend = 4
        else:
            trend = 1
            ep = col[a - b:].max()
            af = 0.02
    elif trend == 4:
        # 首次跌势
        sar = col[a - b:].max()
        if sar < col.max():
            trend = 3
        else:
            trend = 2
            ep = col[a - b:].min()
            af = 0.02
    return sar

def factorMaker59(Data, parameter, pair):
    a = int(parameter['a'])
    b = int(parameter['b'])
    df = Data.copy()
    trend = 3
    df['tmp'] = df['cp_adj_' + pair[0]] / df['cp_adj_' + pair[1]]
    df['sar'] = df['tmp'].rolling(a).apply(sar)
    factor = pd.Series(np.zeros(len(df['sar'])))
    factor[(df['tmp'] > df['sar'])] = 1
    factor[(df['tmp'] < df['sar'])] = -1
    return factor 
    
    
    
    
    
    
    
    
    
    
    
    