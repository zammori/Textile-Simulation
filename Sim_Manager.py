# -*- coding: utf-8 -*-
"""
Created on Mon Dec 23 14:27:02 2024
@author: zammo

Il gestore che organizza la simulazione 
lanciando i vari processi.
Si occupa anche di raccogliere e visualizzare le statistiche. 

"""
from typing import Callable, Optional, Any, Iterable, NamedTuple, Union, Generator
from functools import partial
import simpy as sp
import random
import matplotlib.pyplot as plt
import sys
import math
from itertools import cycle, product
import pandas as pd
import matplotlib.pyplot as plt

from Time_Calendar import Calendar
from Map import City
from Vehicles import Truck
from Bins import Bin
from People import Citizen

Nd = tuple[int, int] # tipo nodo
time = str|int # la data come stringa 

NestedDict = dict[str, Union[str, float,'NestedDict']]

def flat_dct(nst_dct:NestedDict, flat_lst:Optional[list[str|float]] = None, 
                                 not_last:bool = False) -> Generator[list[int|float], None, None]:
    """ trasforma un dizionario in un formato consono ad essere trasformato in Data Frame"""
    for key, val in nst_dct.items():
        try: f_lst = flat_lst[:]
        except: f_lst = []
        if type(val) != dict: 
            f_lst += [key, val]
            yield f_lst 
        else:
            next_values = list(val.values())
            if not_last and (all(type(x) in (int, float) for x in next_values)):
                f_lst += [key] + next_values
                yield f_lst
            else:
                f_lst.append(key)
                for lst in flat_dct(val, not_last = not_last):
                    f_lst2 = f_lst[:] + lst
                    yield f_lst2
        
def make_df(nst_dct:NestedDict, keys:Iterable[str], not_last:bool = False) -> pd.DataFrame:
    records = [dict([(key, val) for key, val in zip(keys, vals)]) for vals in flat_dct(nst_dct, not_last = not_last)]
    df = pd.DataFrame(records)
    return df 

def safe_division(x, y):
    try: return round(x/y, 5)
    except ZeroDivisionError: return 0

class Citizens_Stat():
    
    """ classe che gestisce le statistiche associate alle persone
        specialmente green_awareness  a livello di singoli nodi, poi per quartiere e per città, andando
        ad aggregare """
        
    gaw_trend =('Block', 'Cz_type','Day', 'Gaw')
    ctz_stats = ('Block', 'Cz_type', 'tot_kg', 'tot_rkg', 'tot_lost','tot_km', 'tot_trip', 'n_recycle', 'n_tried_recycle', 'n_full')  
    ctz_dgaw = ('Block', 'Cz_type', 'E_Serv','E_Infl', 'E_Inct', 'E_Dist')
        
    def __init__(self, node_pop:dict[Nd, dict[float: int]], cz_kinds:Iterable[str]):
        self.czk = cz_kinds # tipi di cittadini
        self.nd_p = node_pop # per ogni nodo il numero di cittadini per tipologia!!!
        # il dizionario con la gaw cumulata (somma su tutti i cittadini) per nodo tipo di cittadino e istante temporale
        self.cum_gaw_dict:dict[Nd,dict[str, dict[time, float]]] = {nd:{czk:{} for czk in self.czk} for nd in self.nd_p.keys()} 
        self._all = cz_kinds[-1] # l'ultimo valore contiene l'etichetta che rappresenta tutti i tipi di cittadini
        # il dizionario con i valodi cumulati delle statistiche per nodo e tipo di cittadino
        self.ctz_stat_dict:dict[Nd,dict[str, dict[str, float]]] = {nd:{czk:{st:0 for st in self.ctz_stats[2:]} for czk in self.czk} for nd in self.nd_p.keys()} 
        
        # il dizionario con i valodi cumulati delle variazioni di gaw 
        self.ctz_dgaw_dict:dict[Nd, dict[str, dict[str,float]]] = {nd:{czk:{st:0 for st in self.ctz_dgaw[2:]} for czk in self.czk} for nd in self.nd_p.keys()} 
    
    def update_cum_gaw(self, nd: Nd, czk: str, t:time, gaw:float,*, agg_ctz:bool = False):
        """ aggiorna il dizionario annidato delle gaw cumulate per singolo nodo 
        se agg_ctz True allora si ottiene anche il totale generale """
        czks = (czk, self._all) if agg_ctz else (czk, ) # serve per aggiornare anche a livello globale
        for c in czks:
            dct = self.cum_gaw_dict[nd][c]
            old_gaw = dct.get(t, 0) 
            dct[t] = round(gaw + old_gaw, 5)
            
    def average_gaw(self, nodes:Optional[dict[str, Iterable[Nd]]] = None, cz_kinds:Optional[Iterable[str]] = None) -> dict[str, dict[time, float]]:
        """ restituisce un dizionario con la stessa struttura, ma con valori medi 
            calcolati per nodo, per insieme di nodi, per tipo di popolazione ecc."
            nodes è un dizionario con l'etichetta (es. anello 1) e i nodi associati '
            esempio di chiamata per ottenere gaw degli eco e quella media totale in due aree
            St.average_gaw(nodes = {'area1':((0,0),(0,1)), 'area2':((1,1)(2,2))}, cz_kinds = ('eco', 'all'))
        """
        if not nodes: nodes = {nd:(nd, ) for nd in self.nd_p.keys()}
        if not cz_kinds: cz_kinds = self.czk 
        avg_dct = {bl:{czk:{} for czk in cz_kinds} for bl in nodes.keys()}
        for block, czk in product(nodes.keys(), cz_kinds):
            times = ()
            nds = [nd for nd in nodes[block] if self.nd_p[nd][czk] >0] # solo nodi popolati!!!
            n_ctz = sum(self.nd_p[nd][czk] for nd in nds)
            if nds == []: # se quel tipo di cittadino non c'è lo eliminiamo dal dizionario
                del avg_dct[block][czk]
                continue
            times = tuple(self.cum_gaw_dict[nds[0]][czk].keys()) #cerchiamo i tempi nel primo nodo popolato, tanto i tempi di registrazione sono tutti uguali """
            for t in times:
                cum_val = sum(self.cum_gaw_dict[nd][czk][t] for nd in nds)
                avg_dct[block][czk][t] = safe_division(cum_val, n_ctz) 
             
        return avg_dct
                
    def plot_gaw(self, nodes: Optional[Iterable[Nd]] = None):
        """ plotta i valori medi nel tempo, per tipo e totali, per settore o a livello 
            d'intera mappa """ 
        if not nodes: nodes = tuple(self.nd_p.keys())
        dizionario_dati = self.average_gaw({'area':tuple(nodes)}, cz_kinds = None) # tutte le tipologie comprese all
        dati = dizionario_dati['area'] # prendiamo solo i valori
        x_values = []
        for cz_type, date_gaw in dati.items():
            if not x_values:
                n_dati = len(date_gaw)
                x = list(range(0, n_dati))
            plt.plot(x, list(date_gaw.values()), label = cz_type)
        plt.xlabel('i-th green awareness update')
        plt.ylabel('Green Awareness')
        plt.legend()
        plt.show()        
    
    def create_ctzs_stat(self, citizens:Iterable, delta_gaw: bool = False) -> dict[dict[str, [str, float]]]:
         """ crea il dizionario contente le principali statistiche dei cittadini, 
             a livello cumulato, nodo per nodo. Suddivide per tipologia di cittadino. 
             citizens è un iterabile che restituisce un cittadino alla volta 
             se delta_gaw = True crea il dizionario con i cambi di gaw per i vari effetti"""
         
         def update(dct, val):
             if not math.isnan(val) and val is not None: 
                 dct[cz.nd][cz.att.kind][att] += round(val, 3)
                 dct[cz.nd][self._all][att] += round(val, 3) # mettiamo anche il totale
             
         for cz in citizens:
             if not delta_gaw:
                 for att in self.ctz_stats[2:]: update(dct = self.ctz_stat_dict, val = getattr(cz, att))
             else:
                 delta_gaw = cz.gr_aw_mod(labels = self.ctz_dgaw[2:]) # calcola il totale dei vari effetti
                 for att, val in delta_gaw.items(): update(dct = self.ctz_dgaw_dict, val = val)
              
    def average_ctz_stat(self, nodes:Optional[dict[str, Iterable[Nd]]] = None, cz_kinds:Optional[Iterable[str]] = None, 
                         Cumulative:bool = False,
                         dgaw:bool = False) -> dict[str, dict[str, float]]:
        """ restituisce un dizionario con la stessa struttura, ma con valori medi 
            calcolati per nodo, per insieme di nodi, per tipo di popolazione ecc.
            Cumulative non fa la media per cittadino, ma restituisce i valori cumulati 
        """
        if not nodes: nodes = {nd:(nd, ) for nd in self.nd_p.keys()}
        if not cz_kinds: cz_kinds = self.czk 
        stat_dict = self.ctz_stat_dict if not dgaw else self.ctz_dgaw_dict
        labels = self.ctz_stats[2:] if not dgaw else self.ctz_dgaw[2:]
        
        avg_dct = {bl:{czk:{} for czk in cz_kinds} for bl in nodes.keys()}
        for block, czk in product(nodes.keys(), cz_kinds):
            nds = [nd for nd in nodes[block] if self.nd_p[nd][czk] > 0] # solo nodi popolati!!!
            n_ctz = sum(self.nd_p[nd][czk] for nd in nds)
            if n_ctz <= 0:  # come prima, se questo tipo di cittadino non c'è lo togliamo
                del avg_dct[block][czk]
                continue
            dc = {lb:sum(stat_dict[nd][czk][lb] for nd in nds) for lb in labels}
            if not dgaw: dc['s_level'] = 1 - safe_division(dc['n_full'], dc['tot_trip'])
            if not Cumulative: dc = {st:safe_division(val, n_ctz) for st, val in dc.items() if st != 's_level'}
            avg_dct[block][czk] = dc
        return avg_dct
    
    def crea_data_frame(self, nst_dct: NestedDict, seed:Optional[int] = None, idx_lbl:int = 0, not_last:bool = True) -> pd.DataFrame:
        """ con not last True elenca le statistiche per riga, con false per colonna,
            serve solo per le statistiche, non per il trend della gaw """
        labels = (self.gaw_trend, self.ctz_stats, self.ctz_dgaw) 
        if idx_lbl == 0: not_last = False
        df =  make_df(nst_dct = nst_dct, keys = labels[idx_lbl], not_last = not_last)
        if seed is not None: df['seed'] = seed 
        return df


class Truck_Bin_Stat():
    tr_stats = ('Idx', 'n_trips', 'n_calls', 'tot_km', 'avg_n_sub_tr', 'avg_time','avg_km', 'avg_usage')  
    bin_stats = ('Idx', 'n_call','n_empty', 'n_as_empty', 'n_over_th', 'n_full', 'avg_level', 'n_go', 'kg_tot', 'kg_rlost')
    
    def __init__(self, Tr_B:dict[Truck, tuple[Bin]]):
        self._tr = tuple(Tr_B.keys())
        bins = set()
        for b in Tr_B.values():
            bins = bins.union(set(b))
        self._bn = sorted(tuple(bins), key = lambda b: b.nd)
        # self._tr_stat: dict[int, dict[str, float]] = {}
        self._bn_stat: dict[int, dict[str, float]] = {}
    
    def  crea_data_frame(self, seed:Optional[int] = None, ctz:Optional[Iterable] = None) -> pd.DataFrame:
         """ statistiche riassuntive di tutti i truck o di tutti i bins"""
         if not ctz:
             labels = self.tr_stats
             stat = {idx:{att:getattr(tr, att) for att in labels[1:]} for idx, tr in enumerate(self._tr)}
         else:
             labels = self.bin_stats
             stat = self.make_bins_stat(citizens = ctz)
        
         df = make_df(nst_dct = stat, keys = labels, not_last = True) 
         if seed is not None: df['seed'] = seed 
         return df
    
    def _get_bin(self, bin_idx:Optional[int] = None, bin_nd:Optional[Nd] = None) -> Bin|tuple[Bin]:
         """ ricerca o per nodo o per indice, 
                 si suppone che ogni truck visiti bidoni distinti
                    se entrambi gli argomenti sono None li restituisce tutti """
         if bin_idx is not None: return self._bn[bin_idx]
         if bin_nd is not None:
             for b in self._bn: 
                 if b.nd == bin_nd: return b
         return self._bn

    def make_bins_stat(self, *bin_idx:int, citizens:Iterable) -> dict[int, dict[str, float]]:
         """ statistiche riassuntive di tutti o di alcuni bin
                     totale conferito, extra non conferito   
                            numero di volte pieno, ecc. """
         
         def _is_in(val:int) -> bool:
             """ true se il bin interessa """
             if bin_idx == (): return True # se tupla vuota vanno bene tutti
             return val in bin_idx
         
         # i bidoni che ci interessano
         bins = tuple(b for idx, b in enumerate(self._get_bin()) if _is_in(idx)) 
         # le statistiche di ogni bin d'interesse
         b_stat = {}
         for b in bins:
             b_stat[b.nd] = {key:round(stat, 4) for key, stat in b.stat.items() if key != 'avg_level'}
             try: b_stat[b.nd]['avg_level'] = round(sum(b.stat['avg_level'])/len(b.stat['avg_level']),4)
             except: b_stat[b.nd]['avg_level'] = math.nan
         # aggiungiamo le statistiche di conferimento:
         conf_stat = self._bin_kg(citizens) 
         for bin_nd, node_cstat in conf_stat.items():
             if bin_nd in b_stat.keys(): b_stat[bin_nd].update(node_cstat)
         return b_stat
     
    def _bin_kg(self, citizens:Iterable):
         """ kg conferiti e kg che avrebbero potuto essere conferiti 
                 ad ogni bin """
         conf_kg = {} # conterrà un dizionario bin_stat per ogni bin
         for cz in citizens:
             for bin_node, node_stat in cz.used_bins.items(): # used_bins:dict[Nd, dict[str, float]]   es. (1, 1): {'n_go': 7, 'kg_tot': 8.03, 'kg_rlost': 0},
                 bin_stat = conf_kg.setdefault(bin_node, {key:0 for key in node_stat.keys()}) 
                 conf_kg[bin_node]= {key:round(val + node_stat[key], 2) for key, val in bin_stat.items()}
         return conf_kg
     
    def show_bin_trend(self, idx_bin:int, coor_bin:Optional[Nd] = None, 
                                     t_start:int = 0, t_end:int  = math.inf):
          """ mostra il grafico dell'andamento del riempimento di un bin """
          b:Bin = self._get_bin(idx_bin, coor_bin)
          b.show_pattern(t_start, t_end)
        
class Manager:
    def __init__(self, env, the_city: City, trucks:Iterable[Truck], 
                 ctz_types:Iterable[str], all_ctz = 'all', calendar:Optional[Calendar] = None, 
                         ):
        """ I truck devono già essere stati assegnati ai nodi! """
        self.env = env
        self.ctz_tps = ctz_types
        self.city:City = the_city
        self.tr_bin: dict[Truck, tuple[Bin]] = {tr:tuple(tr.bins.values()) for tr in trucks}
        node_pop = {nd:
                       {ck:self.city.n_citizens(nodes = (nd, ), flt = lambda ctz: ctz.att.kind == ck) 
                           for  ck in self.ctz_tps}
                                  for nd in self.city.map.all_ngh.keys()}
        for nd in node_pop.keys(): # aggiungiamo il totale
            node_pop[nd][all_ctz] = self.city.node_pop(nd)
        
        self.calendar = calendar
        self.ctz_stat = Citizens_Stat(
                                    node_pop = node_pop,
                                    cz_kinds = self.ctz_tps + (all_ctz,)
                                    ) # all_ngh dizionario nodo:vicinato
        self.tb_stat = Truck_Bin_Stat(self.tr_bin)
        
    def __repr__(self):
        return f'{self.city}, n_truck = {len(self.tr_bin)}, n_bins = {sum(len(b) for b in self.tr_bin.values())}'    
        
    """ I Processi Simpy """
    
    def gen_waste_single_citizen(self, mean_time:int, rnd_bin:bool = True):
        """ lancia il processo di generazione e smaltimento rifiuti 
                    per ogni singolo cittadino """
        for cz in self.city.iter_citizens():
            self.env.process(cz.pr_confere_waste(mean_time, rnd_bin))
    
    """ PER ORA QUESTO PROCESSO NON GESTISCE IL CALENDARIO """
    def gen_waste_all_citizen(self, mean_time:int, rnd_bin:bool = True):
        """ processo eterno che fa spostare un cittadino alla volta,
            in modo da non avere enne_mila processi paralleli da gestire """
        all_citizens = tuple(self.city.iter_citizens())
        mean_time = mean_time/len(all_citizens)
        while True:
            cz = random.choice(all_citizens)
            dt = round(random.expovariate(lambd = 1/mean_time), 4)
            yield self.env.timeout(dt)
            cz.confere_waste(rnd_bin)
    
    def apply_incentives(self, tempo_incentivi:Iterable[float]):
        """ processo che attiva gli incentivi, se c'è incentivo iniziale 
            quello è già considerato nella creazione dei cittadini!
            tempo_incentivi contiene gli istanti (in termini assoluti in cui scatta un incentivo),
            gli incentivi sono già assegnati ai tipi di cittadini.
        """
        times = tuple(tempo_incentivi)
        times0 = (0, ) + times[:-1]
        for t, t0 in zip(times, times0):
            yield self.env.timeout(t - t0)
            """ aggiorna la gaw di ogni cittadino in base agli incentivi"""
            for cz in self.city.iter_citizens():
                if (incentive := cz.att.find_incentive(t)): 
                    delta_gaw = cz.gr_aw *(round(incentive(), 3))
                    cz._inc_delta_gaw.append(delta_gaw)
                    cz.gr_aw += delta_gaw # eventualmente aumentata
                    cz._check_gaw()
                    
    def update_green_awareness(self, dt:int|tuple, min_dist:float, wgh: bool = True):
        """ Processo periodico di aggiornamento della green awareness
                se wgh è true si usa l'inverso della distanza come elemento di peso,
                    min dist è la distanza minima che viene utilizzata nel calcolo """
        try: intertempo = cycle(dt)
        except: intertempo = cycle((dt, ))
        while True:
            """ Il primo valore è quella a tempo zero. Poi seguono le altre """
            dt = next(intertempo)
            for cz in self.city.iter_citizens():
                # calcola la average green awareness dei vicini di ogni cittadino
                cz._avg_gaw = cz.compute_av_gaw(wgh = wgh, min_dist = min_dist) 
                # registra in senso cumulato la average green awareness di ogni cittadino
                t = self.env.now
                if self.calendar: t = repr(self.calendar)
                self.ctz_stat.update_cum_gaw(nd = cz.nd, czk = cz.att.kind, t = t, gaw = round(cz.gr_aw,5), agg_ctz = True)

            yield self.env.timeout(dt)
            
            # esegue l'aggiornamento della green awareness di ogni cittadino
            for cz in self.city.iter_citizens():
                cz.update_gaw()
    
    def schedule_trucks(self, Tr:Truck, dt:int):
        """ Processo di gestione dei truck 
                va istanziato per ogni truck utilizzato nella simuazione! """
        corrected_dt = dt
        while True:
            intertempo_fisso = self.env.timeout(corrected_dt)
            segnali = [B.tr for B in Tr.bins.values()] # gli eventi segnale
            # parte o a intervallo fisso, o su segnalazione degli smart bin
            evento_triggher = yield sp.events.AnyOf(env = self.env, events = [intertempo_fisso] + segnali)
            start_time = self.env.now
            # si richiede la risorsa (truck) per evitare che possa partire prima di aver ultimato il giro
            with Tr.request() as req:
                on_call = False if intertempo_fisso in evento_triggher else True
                yield self.env.process(Tr.collect_from_bins(on_call))
            # si registra il tempo che manca alla prossima chiamata
            if self.env.now - start_time >= dt: corrected_dt = 0
            else: corrected_dt = round(dt - (self.env.now - start_time), 4)
    
    """ serve per vedere l'avanzamento della simulazione """
    def progress(self, tot_time:int, perc:float = 0.01):
        n_print = 1/perc
        dt = tot_time / n_print
        pc = perc
        while True:
            yield self.env.timeout(dt)
            print(f'{round(self.env.now,0)} - {perc:.2%}') 
            perc += pc
            
    def _get_bin(self, bin_idx:Optional[int] = None, bin_nd:Optional[Nd] = None) -> Bin|tuple[Bin]:
         """ ricerca o per nodo o per indice, 
                 si suppone che ogni truck visiti bidoni distinti"""
         bins = tuple()
         for bs in self.tr_bin.values(): 
             if bin_nd is not None:
                 for b in bs:
                     if b.nd == bin_nd: return b
             bins += bs
         if bin_idx is not None: return bins[bin_idx]
         return bins
    """ INUTILE EVENTUALMENTE IL COSTO SI CALCOLA DA PARTIRE DAI DATI 
    def tot_cost(self, truck_amm:float, bin_amm:float, euro_km:float, euro_kg:float) -> float:
         #banale funzione di costo del sistema 
         n_truck = len(tuple(self.tr_bin.keys()))
         n_bin = len(self._get_bin())
         r_kg = sum(cz.tot_rkg for cz in self.city.iter_citizens())
         tot_km = sum(tr.tot_km for tr in self.tr_bin.keys())
         return truck_amm*n_truck + bin_amm*n_bin + r_kg*euro_kg + tot_km*euro_km  
   """            
          
            
                
