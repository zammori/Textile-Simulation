# -*- coding: utf-8 -*-
"""
Created on Wed May 14 16:52:01 2025

@author: zammo
"""

""" Calendario """
import simpy as sp
from itertools import cycle

class Calendar():
    def __init__(self, env):
        self.env = env
        self.date = {'yy':1, 'mm':1, 'dd':1}
    
    def make_calendar(self, show:bool = False):
        days = cycle((31,28,31,30,31,30,31,31,30,31,30,31))
        while True:
            n_days = next(days)
            for day in range(2, n_days + 2):
                print(self)
                yield self.env.timeout(24)
                self.date['dd'] = day if day <= n_days else 1
            self.date['mm'] +=1
            if self.date['mm']  == 13:
                self.date['mm'] = 1
                self.date['yy'] += 1
    
    def __repr__(self) -> str:
        return f'{self.day}.{self.month}.{self.year}'       
    
    @property
    def day(self):
        return self.date['dd']
    
    @property
    def month(self):
        return self.date['mm']
    
    @property
    def year(self):
        return self.date['yy']

    def __str__(self):
        return f'DATA - Giorno {self.day} - Mese {self.month} - Anno {self.year}'
    
    def __repr__(self) -> str:
        return f'{self.day}.{self.month}.{self.year}' 