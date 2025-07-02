# -*- coding: utf-8 -*-
from __future__ import annotations
"""
Created on Tue Dec 17 15:45:27 2024
@author: zammo

La classe che rappresenta i cassonetti
per la raccolta dei rifiuti tessili urbani
"""
from typing import Callable, Optional, Any, Iterable
import matplotlib.pyplot as plt 
import simpy as sp
import math

from Time_Calendar import Calendar
Nd = tuple[int, int] # tipo nodo
ev = sp.Event

class Bin:
    """ Smart Bins 
        sono presenti due livelli
            il primo thr fa scattare il segnale di chiamata
            il secondo l_thr è un livello basso che fa considerare il bidone come se fosse vuoto 
    """
    def __init__(self, env, capacity:float, threshold:Optional[float] = None, low_treshold: float = 0,
                         ref_nd:Optional[Nd] = None, coord:Optional[tuple[float, float]] = None,
                         calendar:Optional[Calendar] = None,
                         full_percent = 1): # la percentuale che fa registrare il bin come pieno solo a livello di statistiche
        self.env = env
        self.nd:Nd = ref_nd # il nodo di riferimento
        self.crd:tuple[float, float] = coord # la coordinata cartesiana
        self.lvl:float = 0 # il livello effettivo
        self.cap:float = capacity # istantaneo 
        self.thr:float = threshold if threshold is not None else math.inf # il livello che fa partire il segnale di chiamata
        self.l_thr = low_treshold # il livello che non fa tornare indietro il camion (cassonetto considerato vuoto)
        self.tr = self.env.event() # evento generico
        self.ss:dict[float, float] = {} # la serie storica del livello istantaneo
        self.fp:float = full_percent
        
        self.stat:dict[str, int] = {'n_call':0, 'n_empty': 0, 'n_as_empty':0, 'n_over_th':0, 'n_full':0, 'avg_level': []}
        self.calendar = calendar
    
    """ Proprietà e metodi di visualizzazione """
    
    def __repr__(self):
        return f'[cap:({self.cap}, {self.thr}), at:{{{self.nd}, {self.crd}}}]'
    
    
    def show_pattern(self, start_time:int, end_time:int):
        """ mostra l'andamento del riempimento """
        xy = tuple((k, v) for k,v  in self.ss.items() if k >= start_time and k <= end_time)
        x = tuple(k[0] for k in xy)
        y = tuple(k[1] for k in xy)
        plt.plot(x, y, color='blue')
        plt.xlabel('Time')
        plt.ylabel('Level')
        plt.title(f'Trend of bin at {self.nd}')
        plt.show()
     
    @property
    def is_fully_empty(self) -> bool:
        return self.lvl == 0
    
    @property
    def is_empty(self) -> bool:
        """ sotto il livello low """
        return self.lvl <= self.l_thr 
   
    @property
    def is_over_threshold(self) -> bool:
        return self.lvl >= self.thr
    
    @property
    def is_full(self) -> bool:
        return self.lvl >= self.cap*self.fp
    
    
    """ Proprietà e metodi per la creazione di bin secondo un certo pattern """
    
    def gen_n_at(self, nodes:Iterable[Nd]) -> list[Bin]:
        """ Crea enne bidoni sugli enne nodi passati in input """
        return [Bin(env = self.env, capacity = self.cap, threshold = self.thr, 
                        low_treshold = self.l_thr, ref_nd = nd, coord = None, full_percent = self.fp) for nd in nodes]   
    
    def gen_a_croce(self, lw:int, md:int, hg:int) -> list[Bin]:
        """ dispone i bidoni a croce 
                low è la coordinata inferiore, 
                        md media, hg quella alta """
        nodes = ((lw, md),(md, lw),(md, md),(md, hg),(hg, md))
        return self.gen_n_at(nodes)
     
    
    """ Proprietà e metodi per riempire e per svuotare il bidone """
    
    def collect(self, kg:float) -> float:
        """ toglie i kg passati in ingresso, se disponibili,
                in caso contrario tutto quanto """ 
        self.stat['avg_level'].append(self.lvl) # registriamo il livello che c'era quando abbiamo svuotato il bidone 
        kg = min(kg, self.lvl) # kg sono quelli richiesti, in pratica la capacità disponibile del van
        # codice modificato
        self.lvl -= kg
        if self.is_fully_empty: self.stat['n_empty'] += 1
        elif self.is_empty:  self.stat['n_as_empty'] += 1
     
        # se il livello scende sotto il thr, 
        # ricrea/rinnova l'evento generico di simpy
        if self.lvl < self.thr and self.tr.triggered: 
            self.tr = self.env.event()
        self.ss[round(self.env.now, 4)] = round(self.lvl, 4) # lo storico del livello
        return kg
        
    def put(self, kg:float) -> bool:
        """ prova a mettere i kg passati in ingresso nel bin,
                se non ci stanno non ci mette nulla 
                    se carica restituisce True, altrimenti False """
        tot = self.lvl + kg # nuovo livello teorico
        # anche se il deposito non va a buon fine 
        # ma il bin è quasi pieno si trigghera l'evento segnale!!!
        
        if tot >= self.thr and not self.tr.triggered:  
            """ chiamiamo anche nel caso, raro, che la roba non ci stia """
            self.tr.succeed() 
            self.stat['n_call'] += 1
        
        if tot <= self.cap:
            was_full = self.is_full
            was_over_th = self.is_over_threshold
            self.lvl = tot
            if self.is_full and not was_full: self.stat['n_full'] += 1
            elif self.is_over_threshold and not was_over_th: self.stat['n_over_th'] += 1
            self.ss[round(self.env.now, 4)] = round(self.lvl, 4)
            return True # c'era spazio!!!
        return False # non c'era spazio!!!
    
    """
    def _update_stat(self, kg:int):
        #aggiornato sia al deposito sia al prelievo 
        new_level = self.lvl + kg # kg negativo in caso di preliveo
        st = self.stat
        # non era già vuoto ed è andato a zero (la condizione not fully_empty non serve ...)
        if not self.is_fully_empty and new_level == 0: st['n_empty'] += 1
        elif not self.is_empty and new_level <= self.l_thr:  st['n_as_empty'] += 1
        elif not self.is_full and new_level >= self.cap*self.fp: st['n_full'] += 1
        elif not self.is_over_threshold and new_level >= self.thr: st['n_over_th'] += 1
    """    

""" Classe usata solo per debug """

class Gestore_Prova:
    def __init__(self, env, th = None):
        self.env = env
        self.bins = [Bin(self.env, (0,0), 200, None), 
                     Bin(self.env, (0,1), 100, th), 
                     Bin(self.env, (0,2), 200, None)]
    def deposita(self):
        while True:
            yield self.env.timeout(1)
            for i in range(3):                
                self.bins[i].put(20)
    
    def prelievo_fisso(self):
        while True:
            yield self.env.timeout(15)
            for i  in range(3):
                self.bins[i].collect(200) # svuota tutto
    
    def prelievo_su_chiamata(self):
        while True:
            self.events = [b.tr for b in self.bins]
            risultato = yield sp.events.AnyOf(self.env, self.events)
            print(f'Riceve il segnale alle {self.env.now}')
            for i  in range(3):
                self.bins[i].collect(200) # svuota tutto

"""
Per Debug 
"""        
            
            
""" standard a tempo fisso        
env = sp.Environment()   
Gp = Gestore_Prova(env)
env.process(Gp.deposita())
env.process(Gp.prelievo_fisso())
env.run(until = 60) """

""" a chiamata       
env = sp.Environment()   
Gp = Gestore_Prova(env, th = 90)
env.process(Gp.deposita())
env.process(Gp.prelievo_su_chiamata())
env.run(until = 60) """


                
            
        
        