# -*- coding: utf-8 -*-
from __future__ import annotations
"""
Created on Fri Dec 20 19:04:18 2024

@author: zammo
"""
from typing import Callable, Optional, Any, Iterable, NamedTuple, Generator
from math import nan
import simpy as sp

from Map import DMatrix
from Bins import Bin
from Stats import Trip, T_stat
from Time_Calendar import Calendar

Nd = tuple[int, int] # tipo nodo

class Truck(sp.Resource):
    def __init__(self, env, the_map:DMatrix, capacity:float, threshold: float, velocity:float, 
                     n_in:Nd = None, n_out:Nd = None, bins:Optional[Iterable[Bin]] = None, 
                     t_load: Callable[float, float] = lambda kg: 120, 
                     t_unload: Callable[float, float] = lambda kg: 300,
                     calendar:Optional[Calendar] = None):
        super().__init__(env)
        self.env = env
        self.map:DMatrix = the_map
        self.cap:float = capacity
        self.thr:float = threshold # la quantità che lo fa andare direttamente allo scarico
        self.vel:float= velocity
        self.lvl:float = 0 # il livello effettivo
        self.tld:Callable = t_load
        self.tul:Callable = t_unload
        self.n_in = n_in
        self.n_out = n_out
        # bidoni e routing, se passati, altrimenti vuoti 
        self.bins:dict[Nd, Bin]  = {}
        self.f_route:list = [] # il routing completo
        if bins is not None: self.add_bins(bins)
        # statistiche 
        self.sts:dict[int, T_stat] = {}
        self.n_trip = 0 # numero totale di viaggi
        self.calendar = calendar
    
    """ metodi di creazione """
    def add_bins(self, bins:Optional[Iterable[Bin]] = None):
        """ aggiunge i bins al truck """
        if bins is None:
            self.bins = {}
            self.f_route = []
        else:
            self.bins = {B.nd:B for B in bins}
            self.f_route = [self.n_in] + [nd for nd in self.bins.keys()] + [self.n_out] 
    
    def copy(self, bins:Optional[Iterable[Bin]] = None) -> Truck:
        """ copia il truck, assegnando i bin passati in input """
        return Truck(env = self.env, the_map = self.map, capacity = self.cap, threshold = self.thr, velocity = self.vel, 
                     n_in = self.n_in, n_out = self.n_out, t_load = self.tld, t_unload = self.tul, bins = bins)
    
    def __iter__(self) -> Generator[T_stat, None, None]:
        for ts in self.sts.values():
            yield ts
    
    """ metodi di rappresentazione e proprietà """
    
    def __repr__(self) -> str:
        return f'cap:{self.cap}, f_route = {self.f_route}'
      
    @property
    def is_full(self) -> bool:
        return self.lvl >= self.thr

    """ Proprietà  per statistiche """
    
    @property
    def n_trips(self) -> int:
        """ viaggi fatti """
        return self.n_trip
    
    @property
    def tot_km(self) -> int:
        return sum(ts.km for ts in self)
    
    @property
    def n_calls(self) -> int:
        """ viaggi a chiamata """
        return sum(ts.on_call for ts in self)
    
    @property
    def max_n_sub_tr(self) -> int:
        """ massimo numero di sub-tour """
        return max(self.n_sub_trips)
    
    @property
    def avg_n_sub_tr(self) -> int:
        """ numero medio di sub-tour """
        n_st = self.n_sub_trips
        return round(sum(n_st)/len(n_st), 2)
    
    @property
    def n_sub_trips(self) -> tuple[int]:
        return tuple(ts.n_trips for ts in self)
    
    @property
    def avg_time(self):
        return round(sum(ts.tt for ts in self)/self.n_trip, 3)
    
    @property
    def avg_km(self):
        return round(sum(ts.km for ts in self)/self.n_trip, 3)
    
    @property
    def avg_usage(self):
        """ l'utilizzazione media del mezzo """
        return round(sum(ts.saturation(self.cap) for ts in self)/self.n_trip, 3)
    
    """ I processi di carico e scarico """
    
    def collect_from_bins(self, on_call:bool = False):
        """ il truck visita tutti i bin secondo l'ordine assegnato
                se non ha spazio, va allo scarico, e poi torna al bidone 
                    non completamente svuotato e poi riparte.
                             Infine torna al nodo di partenza ."""
        self.n_trip += 1
        Ts = T_stat()
        if on_call: Ts.on_call = True
        full_route = list(zip(self.f_route[:-1], self.f_route[1:])) # le coppie di nodi da visitare
        while True:
            T = Trip(round(self.env.now, 4), date = repr(self.calendar))
            # parte la sotto-missione 
            # che potrebbe essere anche solo 1, se lo spazio di carico è sufficiente
            yield self.env.process(self._sub_route(full_route, T))
            Ts.add_trips(T)
            # ad ogni sub trip vengono eliminati dal routing
            # tutte le coppie di nodi visitati, se len = 1 resta solo da andare 
            # da out ad in
            if len(full_route) <= 1: break
        self.sts[self.n_trip] = Ts
        
    def _sub_route(self, route:list[Nd], T:Trip):
        """ Una sotto missione. Il truck prosegue 
                sino a quando è pieno e deve andare a scaricare.
                   Ad ogni sub_trip vengono tolte le coppie di nodi visitate 
                       se ne mancano il tour viene aggiornato aggiungendo 
                           lo spostamento da out a nodo da visitare """
        while True:
            # eliminiamo il primo ramo dal routing
            # n1, e n2 sono il nodo di partenza ed il nodo di arrivo
            n1, n2 = route.pop(0) 
            # spostamento al bin posizionalto in n2 
            try: B = self.bins[n2]
            except: raise Exception('Il nodo deve avere un bin!!!')
            # tempo di viaggio + aggiornamento statistiche
            yield self.env.process(self._trip(n1, n2, T)) 
            # tempo di carico
            yield self.env.process(self._load(B))
            
            if self.is_full or len(route) == 1: # deve andare all'output o perchè pieno, o perchè ha finito il tour
                
                yield self.env.process(self._go_to_out(n2, T)) # va a out
                
                # caso 1: bin in n2 non svuotato, bisogna tornare lì! """
                if not B.is_empty: 
                    route.insert(0, (self.n_out, n2)) # nuovo ramo da out a n2
                
                # caso 2: bin in n2 svuotato, ma ce ne sono altri da visitare 
                elif B.is_empty and len(route) > 1: 
                    route[0] = (self.n_out, route[0][1]) # nuovo ramo da out a nx, con nx il bin successivo che si legge in route[0][1]
                
                # caso 3: fine giro, hurra!!!
                else: 
                    yield self.env.process(self._go_to_in(self.n_out, T)) # va a input
                
                T.t_end = round(self.env.now, 4)
                break
            
            else: # inutile, ma serve solo a segnalare che il ciclo riprende 
                continue # il ruck prosegue il tour
                    
    def _trip(self, n1:Nd, n2:Nd, T:Trip) -> float:
        """ il processo di spostamento singolo """
        dist = self.map[n1, n2] # la distanza
        T.tr.append((n1, n2))
        T.km += dist
        yield self.env.timeout(round(dist/self.vel), 4)
    
    def _load(self, B:Bin):
        """ il processo di carico """
        kg = self.cap - self.lvl # valore massimo che può essere caricato
        kg = B.collect(kg) # valore effettivo del caricato
        yield self.env.timeout(self.tld(kg)) # tempo di carico
        self.lvl += kg
    
    def _go_to_in(self, nd: Nd, T):
        """ tempo di ritorno"""
        yield self.env.process(self._trip(nd, self.n_in, T)) # torna alla partenza
        
    def _go_to_out(self, nd:Nd, T:Trip):
        """ tempo di andata ad out + tempo di scarico """
        yield self.env.process(self._trip(nd, self.n_out, T)) # va allo scarico
        yield self.env.process(self._unload(T))
    
    def _unload(self, T:Trip):
        """ scarico ad out, sia a fine ciclo, 
                sia a fine ciclo parziale """ 
        yield self.env.timeout(self.tul(self.lvl)) # tempo di scarico
        T.t_end = round(self.env.now, 4)
        T.kg = self.lvl
        self.lvl = 0
    
    
    
    
""" Per debug 
import networkx as nx
import itertools

def gen_graph(n = 3, d = 100):
    G = nx.grid_2d_graph(n, n)
    for i, j in G.edges():
        G[i][j]['distance'] = d 
    return G  

G = gen_graph()
M = DMatrix({(n1, n2): nx.shortest_path_length(G, source = n1, target = n2, weight='distance') 
                    for n1, n2 in itertools.product(G.nodes(), G.nodes())}) 

env = sp.Environment() 
bins = [Bin(env = env, capacity = 10, threshold = None, ref_nd = nd) for nd in ((0, 2),(1, 1), (1, 2), (2, 1))] 

q1 = (3, )*4 # con questa quantità fa un solo giro
q2 = (2, 2, 2, 10) # con questa quantità fa 2 giri giro_1(B1:2, B2:2, B3:2, B4:9), giro_2(B4:1)
q3 = (10,)*4 # con questa quantità deve fare 3 giri giro_1(B1:10, B2:5), giro_2(B2:5, B3:10), giro_3(B4:10)


Tr = Truck(env = env, capacity = 15, threshold = 15, velocity = 10, n_in = (0,0), n_out = (2,2), bins = bins, the_map = M)

def run_sim(env, bins, qs, nrun = 1):
    for i in range(nrun):
        for b, q in zip(bins, qs): b.put(q)    
        req = Tr.request()
        yield req
        yield env.process(Tr.collect_from_bins())
        Tr.release(req)  
        t = tuple(Tr.sts.keys())[0]
        #breakpoint()
        print(Tr.sts[t].full_route)
    
env.process(run_sim(env, bins, q1, 2))
env.run()
"""
            
            
        
        
