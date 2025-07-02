# -*- coding: utf-8 -*-
from __future__ import annotations

"""
Created on Tue Dec 17 15:21:34 2024
@author: zammo

La classe che modellizza il processo/agente "cittadino".
Ogni cittadino genera rifiuti tessili e sceglie in base alla sua green awareness se 
conferirli alle isole ecologiche o se smaltirli nell'indifferenziato.
La sua propensione si modifica nel tempo sia in base al livello di servizio percepito
sia in base al comportamento dei suoi vicini.

"""

import math
import random
from typing import Optional, Iterable, NamedTuple, Callable, Generator

from Time_Calendar import Calendar
from Bins import Bin
from Stats import Trip

Nd = tuple[int, int]

class C_Trip(Trip):
    """ Classettina per registrare 
        le statistiche dei cittadini """
    def __init__(self, t:float):
        super().__init__(t)
        # il nodo in cui è andato 
        # e che ha trovato pieno o vuoto
        self.tr:Nd = tuple()
        # None se non aveva intenzione di riciclare, 
        # negativo se voleva riciclare, ma ha trovato il bidone pieno
        # il valore negativo è quanto avrebbe conferito se avesse potuto ...
        self.r_kg:float|None = None
        # vero se va a piedi 
        self.bf: bool = False 
        # per debug registriamo anche la variazione di green awareness
        self.var_gaw:float = 0  
        
    def __repr__(self):
        """ kg indifferenziati, kg riciclati, kg che avrebbe riciclato """
        return f'landf.:{self.kg_waste}, rec.:{self.kg_recycled}, r_lost: {self.kg_rlost}'
    
    @property
    def recycle(self) -> bool:
        """ true se ha effettivamente riciclato"""
        # deve essere andato al bin, e averlo trovato vuoto, quindi r_kg positivo 
        return self.tr != () and self.r_kg > 0
    
    @property
    def tried_recycle(self) -> bool:
        """ true se ha provato a riciclare"""
        # deve essere andato al bin, 
        return self.tr != ()
        
    @property
    def kg_waste(self) -> float:
        """ kg nell'indifferenziato """
        return self.kg - self.kg_recycled
    
    @property
    def kg_recycled(self) -> float:
        """ kg reciclati """
        if self.recycle: return self.r_kg
        return 0
    
    @property
    def full_bin(self) -> float|None:
        """ True se il bin era pieno """
        if not self.tried_recycle: return None # è andato al bidone, ma r_kg è negativo 
        return 1 - self.recycle
    
    @property 
    def kg_rlost(self) -> float:
        """ kg che avrebbe differenziato, 
                ma che non ha potuto differenziare """
        if self.full_bin: return abs(self.r_kg)
        return 0

from dataclasses import dataclass, field,fields
@dataclass(unsafe_hash=True)
class Citizen_type:
    """ data class ....
            che contiene tutti gli attributi che definiscono un cittadino.
                Molti sono "probabilistici" e si concretizzeranno per il cittadino specifico """
    
    """ attributi descrittivi """
    kind:str|None = None
    sex:str|None = None
    instruction_level:str|None = None
    income:float|None = None
    s_factor:float = 2.5 # quando c'è stagionalità la quantità è più che doppia
    min_waste:float = 0 # valore minimo generato prima di andare a conferire 
    m_season:tuple[int] = (5, 11) # maggio e novembre
    
    """ parametri comportamentali """
    green_awareness:tuple[float, float]|None = None # il valore puntuale verrà creato come uniforme tra i due valori
    radius_of_influence:tuple[float, float]|None = None
    n_depth:int|None = None
    
    """ distribuzioni di probabilità """
    gen_waste:Callable[[], float]|None = None # kg creati
    green_inc:Callable[[float], float]|None = None # funzione incremento di green awareness in base al livello di servizio, in input gaw
    green_dec:Callable[[float], float]|None = None # funzione decremento di green awareness (come sopra)
    f_influence:Callable[[float], float] = None # funzione passa_parola, influenza tra vicini, in imput delta gaw
    # gli incentivi cambiano nel tempo, per cui predisponiamo una lista d'incentivi nel tempo
    f_incentives: dict[float, Callable[[], float]] = field(default = None, compare = False, hash = False) 
    # le funzioni d'incremento di green awareness in base a incentivo, basso, medio, ecc.
    f_dist_penalty:Callable[[float], float] = None # funzione di decremento della green awareness in base alla distanza dal bidone più vicino, generata una volta sola come per l'effetto dell'incentivo
    f_move:Callable[[float], tuple[True, float]] = None # funzione che determina il tipo (True -> a piedi) e il tempo di spostamento dell'utente (usato solo come statistica)
    
    """ Metodi di supporto """
    
    def find_incentive(self, time) -> Callable|None:
        if (inc:=self.f_incentives) is not None:
            return inc.get(time, None)
      
    def _function_list(self) -> tuple[str]:
        """ lista delle funzioni """
        return tuple(f for f in self.__dataclass_fields__
                                 if callable(getattr(self, f)) and getattr(self, f) is not None)
    
        # vecchio codice con named tuple
        #return tuple(key for key, value in self._asdict().items() if callable(value))
    
    def non_null_rep(self, *taboo:str) -> str:
        """ rappresentazione con solo campi valorizzati """
        taboo += self._function_list() # esclude le funzioni!
        non_null_att = [f'{att} = {getattr(self, att)}' for att  in self.__dataclass_fields__ if (att not in taboo) and (getattr(self, att) is not None)]
        return ", ".join(non_null_att)
    
    def function_desc(self) -> tuple[str]:
        """ lista dei nomi delle funzioni """
        return tuple(getattr(self, foo).__name__ for foo in self._function_list())



# NOTA BENE - COSA EVENTUALMENTE DA SISTEMARE
# LA DIFFERENZA DI GAW TRA AGENTE E VICINATO (USATA PER INFLUENZA RECIPROCA)
# DI DEFAUTL E' CONSIDERATA IN PERCENTUALE 


class Citizen:
    """ una patch per avere rapidamente 
        quantità riciclata e persa 
        anno per anno """
    quantity:dict[str, dict[int, float]]= {'rec':{}, 'lost':{}} 
    
    def __init__(self, env, 
                         xy_coor:tuple[float, float] = (0.0, 0.0), # coordinate x, y
                         reference_node:tuple[int,int] = (0, 0), # nodo di riferimento / quartiere
                         distance_from_node: float = 0.0, # distanza dal nodo di riferimento
                         bins: Optional[dict[Bin, float]] = None, # i bidoni ai quali può andare e la relativa distanza
                         attributes: Optional[Citizen_type] = None, # gli attributi descrittivi
                         influencers: Optional[dict[Citizen, float]] = None, # cittadino e distanza    
                         calendar:Optional[Calendar] = None # il calendario
                         ):
        
        """ Serie di attributi che non vanno mai cambiati!!! 
                ... per pigrizia non avevo voglia di mettere _  @_@ """
        self.env = env # l'ambiente simpy
        self.xy = xy_coor # la posizione cartesiana
        self.nd = reference_node # il nodo di riferimento
        self.ds =  distance_from_node # la distanza dal nodo di riferimento 
        self.calendar = calendar
        # aggiungiamo la distanza tra la posizione della persona 
        # e il nodo di riferimento, e ordiniamo i bidoni per distanza 
        self.bs:dict[Bin, float] = {} # lista dei bin e relativa distanza
        if bins is not None: self._add_bins(bins) 
        # aggiungiamo gli influencers, i vicini che possono influenzare il comportamento 
        self.infs:dict[Citizen, float] = influencers if influencers is not None else {} # i vicini e la loro distanza 
        # gli attributi descrittivi del cittadino
        self.att:Citizen_type  =  attributes 
        self._cum_waste:float = 0 # il valore cumulato che ha generato
        
        """ attributi che vengono calcolati """
        # la green awareness di base viene calcolata usando un'uniforme tra un minimo e un massio
        # tale valore può cambiare se c'è un incentivo ...
        self.gr_aw = round(random.uniform(*self.att.green_awareness), 4) # green awareness calcolata come uniforme
       
        """ Se l'incentivo viene fatto subito allora lo attiviamo direttamente con il costruttore,
            eventuali altri vengono eseguiti in seguito dal sim manager """
        if (incentive := self.att.find_incentive(0)): # cerca l'incentivo a tempo zero
            inc = round(incentive(), 3)*self.gr_aw # triggheriamo il primo incentivo, se esistente
            self.gr_aw += inc
            self._check_gaw()
        else: inc = 0
        
        if self.att.f_dist_penalty is not None and self.bs != {}: self.distance_penalty() # riduce la gaw per effetto della distanza dal bidone più vicino
        self.rof = round(random.uniform(*self.att.radius_of_influence), 4) # il raggio d'influenza calcolato come uniforme
        
        """ attributi aggiunti """
        self.n_trip:int = 0 # numero totale di conferimenti
        self.sts:dict[int:C_Trip] = {} # le statistiche 1:T1, 2:T2, ...
        self._avg_gaw:float = 0.1 # la green awareness media, fissata ad un valore casuale tanto verrà aggiornata
        self._inf_delta_gaw:list[float] = [] # per debug registriamo le variazioni di green awareness per effetto dell'influenza
        self._inc_delta_gaw:list[float] = [inc] # per debug registriamo le variazioni per effetto degli incentivi
        
        """ attributi tolti """
        # self._wgh un altro attributo che viene aggiungo dal metodo _add_bins
        # self._dist_delta_gaw:float = 0 # incluso nel metodo distnace_penalty

    def _add_bins(self, bins: dict[Bin, float]):
        """ aggiunge i bin e la loro distanza (passata in input), 
                alla distanza originaria (calcolata sul grafo)
                    viene aggiunto lo spostamento necessario a 
                        raggiungere il nodo di riferimento """
        self.bs.update({B: d + self.ds for B, d in bins.items()})
        self.bs = dict(sorted((Bd for Bd in self.bs.items()), key = lambda x: x[1])) # ordinati per distanza
        self._wgh = self._make_wgh() # genera i pesi per la selezione
    
    def distance_penalty(self):
        # se c'è l'effetto di penalizzazione in base alla distanza, 
        # aggiorna la green awareness
        penalty = self.att.f_dist_penalty
        closest_bin = tuple(self.bs.keys())[0] # il primo
        min_dist = self.bs[closest_bin]
        delta_gaw =  self.gr_aw * round(penalty(min_dist), 4)
        self._dist_delta_gaw = -delta_gaw
        self.gr_aw -= delta_gaw
        self._check_gaw()
    
    def _make_wgh(self) -> tuple[float]:
        """ pesi inversamente proporzionali alla distanza """
        inv_dst = tuple(1/max(d, 0.1) for d in self.bs.values()) # l'inverso della distanza
        return tuple(round(max(d, 0.1)/sum(inv_dst), 4) for d in inv_dst)
    
    """ Proprietà o funzioni descrittive """    
    def __repr__(self) -> str:
        s = self.att.non_null_rep("green_awareness", "radius_of_influence")
        return f'{s}, rof: {self.rof}, gr_aw: {self.gr_aw}, pos:{self.xy}, node: {self.nd}, infs: {self.n_influencers}'
    
    def __getitem__(self, idx:int) -> C_Trip:
        """ restituisce una statistica di un viaggio specifico """
        return self.sts.get(idx, None)
    
    def __iter__(self) -> Generator[C_Trip, None, None]:
        for tr in self.sts.values():
            yield tr    
    
    @property
    def n_influencers(self) -> int:
        """ numero d'influencers """
        try: return len(self.infs)
        except: return 0
    
    @property
    def show_distrib(self) -> tuple[str]:
        """ il nome delle distribuzioni utilizzate """
        return self.att.function_desc()
    
    @property
    def tot_trip(self) -> int:
        """ numero totale di conferimenti """
        return self.n_trip
    
    @property 
    def tot_kg(self) -> float:
        """ totale generato """
        return sum(trip.kg for trip in self)
    
    @property 
    def tot_rkg(self) -> float:
        """ totale riciclato """
        return sum(trip.kg_recycled for trip in self)
    
    @property 
    def tot_lost(self) -> float:
        """ totale potenzialmente riciclabile """
        return sum(trip.kg_rlost for trip in self)
    
    @property 
    def tot_km(self):
        """ km in auto percorsi """
        return sum(trip.km for trip in self if trip.bf == False)
    
    @property
    def n_recycle(self) -> int:
        """ numero di volte che ha riciclato """
        return sum(trip.recycle for trip in self)
    
    @property
    def n_tried_recycle(self) -> int:
        """ numero di volte in cui è andato al bidone
                per ricilcare, indipedentemetne dal fatto
                    che l'abbia trovato pieno o vuote  """
        return sum(trip.tried_recycle for trip in self)
    
    @property
    def n_full(self) -> int:
        """ numero di volte in cui il bidone era pieno """
        return sum(trip.full_bin for trip in self if trip.full_bin is not None)

    @property
    def s_level(self) -> float:
        """ livello di servizio """
        nr = self.n_tried_recycle
        nf = self.n_full
        # giusto per debug, 
        # controllo su nf e nr
        if nf > nr: 
            breakpoint()
            raise Exception("Valori incongruenti in People, per il calcolo del livello di servizio")
        if nr == 0: return math.nan
        return round(1 - nf/nr, 4)
    
    @property
    def used_bins(self) -> dict[Nd, dict[str, float]]:
        """ numero di volte 
                in cui è andato a ciascun bin 
                    con indicazione del quantitativo conferito """
        stat = {}
        for tr in self:
            if (nd:= tr.tr) != ():
                sub_dict = stat.setdefault(nd, {'n_go' : 0, 'kg_tot' : 0, 'kg_rlost' : 0})
                stat[nd] = {key:val + dv for (key, val), dv in zip(sub_dict.items(), (1, round(tr.kg_recycled, 2), round(tr.kg_rlost, 2)))}
        return stat
    
    def gr_aw_mod(self, labels:tuple[str] = ('eff_servizio', 'eff_influenza', 'eff_incentivi', 'effetto_distanza')) -> dict[str, list[float]]:
        """ mostra le variazioni di green awareness che sono occorse durante la simulazione """
        out = {}
        out[labels[0]] = sum([round(tr.var_gaw, 4) for tr in self])
        out[labels[1]] = sum([round(dg, 4) for dg in self._inf_delta_gaw])
        out[labels[2]] = sum(self._inc_delta_gaw.copy())
        out[labels[3]] = self._dist_delta_gaw
        return out
       
    """ Metodi legati al processo di conferimento dei rifiuti """
    
    def choose_bin(self, rnd = True) -> Bin:
        """ restituisce il più vicino, 
            oppure un bin casuale scelto in base alla distanza """
        idx = 0 if not rnd else random.choices(tuple(range(len(self.bs))), self._wgh, k = 1)[0]
        return tuple(self.bs.keys())[idx]
     
    def _recycle(self) -> bool:
        """ riciclo oppure no? """
        rnd = random.random()
        return rnd <= self.gr_aw
    
    def _seasonality(self) -> float:
        """ se il mese è stagionale restituisce il fattore stagionale
                               altrimenti restituisce 1 """
        if self.calendar is not None and self.calendar.month in self.att.m_season: return self.att.s_factor
        return 1   
    
    def gen_waste(self, rnd_bin = True):
        """ Crea il rifiuto e decide se 
                accumularlo o se conferirla """
        s_factor = self._seasonality()
        self._cum_waste += round(self.att.gen_waste()*s_factor, 2)
        if self._cum_waste >= self.att.min_waste: 
            self.confere_waste(self._cum_waste, rnd_bin)
            self._cum_waste = 0
        
    def confere_waste(self, kg_waste:float, rnd_bin:bool = True):
        """ il conferimento del rifiuto tessile, s_factor è il fattore di 
            stagionalità """
        # print(f'conferimento alle {self.env.now}')
        self.n_trip += 1
        Tr = C_Trip(t = round(self.env.now, 4))
        Tr.kg = kg_waste
        """
        Parte tolta e messa in _gen_waste
        # verifica stagionalità
        if self.calendar is not None:
            if self.calendar.month in self.att.m_season: s_factor = self.att.s_factor
        Tr.kg = round(self.att.gen_waste()*s_factor, 2)
        """
        if self._recycle(): # se ricicla... 
            B = self.choose_bin(rnd_bin) # sceglie il bin
            Tr.tr = B.nd
            Tr.km = 2*self.bs[B]
            k = self.calendar.year
            if B.put(Tr.kg): # trova bidone vuoto
                Tr.r_kg = Tr.kg
                """ patch """
                Citizen.quantity['rec'][k] = Citizen.quantity['rec'].get(k, 0) + Tr.kg
                """ fine patch """
                try: Tr.var_gaw = self.att.green_inc(self.gr_aw) # l'impatto del bidone dipende dalla gaw
                except: Tr.var_gaw = self.att.green_inc() # non dipende
                self.gr_aw += Tr.var_gaw
            else: # trova bidone pieno
                Tr.r_kg = -Tr.kg
                """ patch """
                k = self.calendar.year
                Citizen.quantity['lost'][k] = Citizen.quantity['lost'].get(k, 0) + Tr.kg
                """ fine patch """
                try: Tr.var_gaw = self.att.green_dec(self.gr_aw)
                except: Tr.var_gaw = self.att.green_inc() # non dipende
                self.gr_aw += Tr.var_gaw
            self._check_gaw()
            if self.att.f_move is not None:
                by_foot, tr_time = self.att.f_move(self.bs[B]) # la scelta si basa sulla distanza dal bin self.bs[B]
                Tr.bf = by_foot
                Tr.t_end = Tr.t_str + tr_time
        else: Tr.bf = True # se non ricicla va a piedi, inutile perchè la distanza è comunque nulla
        self.sts[self.n_trip] = Tr # aggiungiamo il report alle statistiche generali
        
    def pr_confere_waste(self, mean_time:int, rnd_bin = True):
        """ il processo simpy se attivato per ogni persona """
        while True:
            time = round(random.expovariate(lambd = 1/mean_time), 4)
            yield self.env.timeout(time)
            self.gen_waste(rnd_bin)
    
    def compute_av_gaw(self, wgh: bool = True, min_dist:float = 1) -> float:
        """ calcola la media pesata della green awareness nel vicinato"""
        if self.infs == (): return self.gr_aw # se non ci sono influencer la media coincide con sè stesso, e quindi il gap sarà nullo
        if wgh: 
            t_gaw, t_inv_dist = 0, 0
            for cz in self.infs:
                inv_dist = 1/max(min_dist, self.infs[cz])
                t_gaw += cz.gr_aw * inv_dist
                t_inv_dist += inv_dist  
            return t_gaw/t_inv_dist
        else: 
           return sum(cz.gr_aw for cz in self.infs)/self.n_influencers
    
    def update_gaw(self, avg_gaw:Optional[float] = None, percentual: bool = True):
        """ ricalcola la green awareness che si modifica
                 per effetto dell'influenza esercitata dai vicini """
        if avg_gaw is None: avg_gaw = self._avg_gaw
        delta_gaw = self.gr_aw - avg_gaw # il delta rispetto alla media pesata della green awareness
        if percentual: delta_gaw /= self.gr_aw
        inc_dec_gaw = self.att.f_influence(delta_gaw) # calcola la probabilità di cambiamento e l'eventuale cambiamento
        self.gr_aw += inc_dec_gaw
        self._inf_delta_gaw.append(inc_dec_gaw)
        self._check_gaw()
    
    def _check_gaw(self):
        if self.gr_aw > 1: self.gr_aw = 1
        if self.gr_aw <= 0: self.gr_aw = 0.0001


""" PER DEBUG 

foo = f_influence(alpha = 0.9, beta = 2.5, delta_inc=0.01, delta_dec = 0.02)
for gap in (-0.5 + i * 0.1 for i in range(11)):
    print(gap, ":", foo(gap))
         
C1 = Citizen_type(kind ='eco_friendly', green_awareness = (0.65, 0.75), radius_of_influence = (200, 300), n_depth = 1,
                  gen_waste = waste_gen(10,30, 40), green_inc = green_variation(delta = 0.01, increase = True),
                  green_dec = green_variation(delta = 0.02, increase = False),
                  f_influence = f_influence(alpha = 0.5, beta = 2.5, delta_inc = 0.01, delta_dec = -0.02))
                    
                      
x = C1.function_desc()

Cz = Citizen(env = None, xy_coor = (10, 12), reference_node = (0, 0), bins = None, attributes = C1, influencers = None)

"""
