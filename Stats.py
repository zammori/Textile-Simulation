from __future__ import annotations
"""
Created on Fri Dec 20 19:04:18 2024
@author: zammo

Una classettina che serve a raccogliere alcuni dati
con finalità statistiche 

"""
from typing import Callable, Optional, Any, Iterable
from math import nan

Nd = tuple[int, int] # tipo nodo

class Trip:
    """ Classe che contiene le statistiche di  
           un viaggio (o di un mezzo o di una persona)"""
    def __init__(self, t:float, date:Optional[str] = None):
        self.date = date
        self.t_str:float = t # tempo inizio
        self.t_end:float  = nan # tempo fine
        self.tr:list[Nd] = [] # i nodi visitati nel giro (trip)
        self.km:float = 0 # i km totali percorsi
        self.kg:float = 0 # i kg totali trasportati
    
    def __repr__(self) -> str:
        return f'Trip: time = {self.tt}, kg = {self.kg}'
    
    @property
    def tt(self) -> float:
        return self.t_end - self.t_str
    
class T_stat:
    """ l'insieme dei viaggi fatti da un mezzo
            per completare il suo routing """
    
    def __init__(self):
        self.trs:list[Trip] = []  # le statistiche di ogni sotto viaggio fatto
        self.on_call: bool = False # vero se viaggio a chiamata
    
    def add_trips(self, *trs:Trip):
        for tr in trs:
            self.trs.append(tr)
    
    def __repr__(self) -> str:
        return f'(on_call: {self.on_call}, Pn_trip: {self.n_trips}, kg: {self.kg}, time: {self.tt})'
    
    def tot_xstat(self, xstat:str) -> float:
        """ restituisce il totale di un attributo xstat passato in input """
        try: return sum(getattr(tr, xstat) for tr in self.trs)
        except: return nan
    
    """ alcune proprietà che sfruttano tot_xstat """
    
    @property
    def n_trips(self) -> int:
        return len(self.trs)
    
    @property
    def kg(self) -> float:
        return self.tot_xstat('kg')
    
    @property
    def tt(self) -> float:
        return self.tot_xstat('tt')
    
    @property
    def km(self) ->float:
        return self.tot_xstat('km')
    
    @property
    def full_route(self) -> list[Nd]:
        fr = []
        for tr in self.trs: fr.extend(list(tr.tr))
        return fr
    
    def saturation(self, max_cap:float) -> float:
        """ la saturazione del mezzo """
        return self.kg/(max_cap*self.n_trips) # quantità trasportata su quantità trasportabile negli enne giri
        
    
   
    
    


        