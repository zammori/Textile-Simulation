# -*- coding: utf-8 -*-
from __future__ import annotations
"""
Created on Tue Dec 17 15:04:26 2024
@author: zammo

Classi che creano e popolano i quartieri
della città

"""

import networkx as nx
import simpy as sp
import random
import math
import itertools
import pandas as pd
import matplotlib.pyplot as plt

from functools import partial
from typing import Callable, Optional, Any, Generator, Iterable, Any

from Time_Calendar import Calendar
from People import Citizen, Citizen_type
from Bins import Bin


""" Definizione tipi variabili """           
f_pop = Callable[[],int] # tipo funzione randomica per creare la popolazione
Nd = tuple[int, int] # tipo nodo
Gr = nx.classes.graph.Graph # tipo grafo
Df = pd.core.frame.DataFrame

"""**** ALCUNE FUNZIONI D'APPOGGIO **** """

def normal_pop(mu:int, sigma:int, min_pop:int = 1):
    """ genera una popolazione gaussiana 
            eventualmente troncando an valore minimo """
    def inner():
        return max(min_pop, int(random.normalvariate(mu, sigma)))
    inner.__name__ = f'normal_pop({mu}, {sigma})'
    return inner

def gen_coord(nr:int, nc:int, dst:float, *, n_decimal:int = 3) -> Generator[tuple[float, float], None, None]:
    """ Genera le coordinate cartesiane dei nodi del network, 
            i nodi devono essere a griglia
            e equispaziati con distanza pari a dst  """
    x, y = 0, 0
    for r in range(nr):
        for c in range(nc):
            yield x, y
            x = round(x + dst, n_decimal)
        x = 0
        y = round(y + dst, n_decimal)

def gen_neighbor(nd:Nd, n_layer:int = 1) ->  Generator[Nd, None, None]: 
    """ Genera la lista dei nodi vicini ad un nodo di partenza, 
            compreso il nodo stesso: es. (1, 1) genera (0, 0), (1, 0), ecc. 
                con n_layer = 0 viene restituito il nodo stesso, 
                        con n_layer = 1 vengono restituiti gli 8 nodi vicini pià il nodo stesso """
    if n_layer == 0: yield nd      
    else:
        x, y = nd
        xmin, xmax = x - n_layer, x + n_layer + 1   
        ymin, ymax = y - n_layer, y + n_layer + 1 
        for ng in itertools.product(range(xmin, xmax), range(ymin, ymax)):
            yield ng
       
def random_circle(center:tuple[float, float], radius:float, *, n_decimal:int = 3)-> tuple[float, float]:
    """ Genera una coordinata cartesiana casuale all'interno di un cerchio """
    teta = random.uniform(0, 2*math.pi)
    r = random.uniform(0, radius)
    x, y = center
    return round(x + r*math.sin(teta), n_decimal), round(y + r*math.cos(teta), n_decimal)

""" **** LE CLASSI **** """

class Neighborhood:
    """ Classe che definisce le specifiche di un quartiere 
            i.e., la percentuale di ciascun tipo di cittadino presente nel quartiere"""
    
    def __init__(self, population: int|Callable[[], float], type_prob:dict[Citizen_type, float]):
        # population è un valore intero o una distribuzione che verrà usata
        # per assegnare la popolazione al quartiere 
        # type_prov è un dizionario che associa alla tipologia del cittadino la probabilità
        # che quel cittadino venga creato 
        if sum(pr for pr in type_prob.values()) != 1: raise Exception('Probabilità errata!')
        if not callable(population): 
            self.pop = partial(lambda x: x,  int(population))
            self._fname = f'{int(population)} fixed'
        else:
            # quando copiamo un vicinato col metodo copy(), population è sempre callable,
            # ma potrebbe essere una lambda parzializzata che restituisce un valore fisso
            # e che non ha nome, in questo caso prima mettiamo self._fname a None e dopo
            # sarà copy a settarlo correttamente.
            try: self._fname = population.__name__
            except: self._fname = None 
            self.pop = population
        self.tpr: dict[Citizen_type, float] = type_prob
    
    def copy(self) -> Neighborhood:
        """ copia del vicinato """
        ng = Neighborhood(population = self.pop, type_prob = self.tpr)
        ng._fname = self._fname 
        return ng
    
    def __repr__(self) -> str:
        return f'population: {self._fname}, kinds: {tuple((cz.kind, prob) for cz, prob in self.tpr.items())}'
    
    def __call__(self, env, center:tuple[float, float] = (0, 0), radius:float = 0, 
                 nd:Nd = (0,0), bins:Optional[Iterable[Bin]] = None,
                 calendar:Optional[Calendar] = None) -> tuple[Citizen]:
        """ crea i cittadini del quartiere 
                piazzandoli all'interno di un'area circolare """
        probabilities = tuple(self.tpr.values())
        c_types = tuple(self.tpr.keys())
        n_ctz = self.pop()
        positions = (random_circle(center, radius) for _ in range(n_ctz))
        return tuple(Citizen(env = env, xy_coor = pos, reference_node = nd, distance_from_node = round(math.dist(center, pos), 4),
                             bins = bins, attributes = random.choices(c_types, probabilities, k = 1)[0],
                             calendar = calendar)  
                     for pos in positions)

def move(dx, dy):
    def inner(x, y): 
        return x + dx, y + dy
    return inner 

def iter_prob(n_dict:dict = {1:5, 2:3, 3:2}):
    """ genera le chiavi tante volte 
        quanto indicato dal corrispondente valore, in modo rando,
        es. 1 1 2 3 1 2 2 1 3 1 """
    dct = n_dict.copy()
    while True:
        rem = sum(dct.values())
        if not rem: return  
        key = random.choices(list(dct.keys()), weights = tuple(dct[k]/rem for k in dct.keys()))[0]
        yield key
        dct[key] -= 1

                            
class All_Neighborhood:
    """ crea tutti i quartieri della città, indicando ad ogni nodo popolato
            quali tipologie di cittadini e in che percentuali ci stanno 
                es. nel nodo x,y il 60% è eco friendly e il 40% è non-eco. 
                    per farlo è necessario usare un metodo di creazione (all_equal, all_random), ecc.
                        oppure farlo manualmente, nodo per nodo usando il metodo di assegnazione [] """
        
    def __init__(self, neighborhoods:Iterable[Neighborhood]):
        """ registra le tipologie di vicinato e predispone
                il dizionario all_ngh che associa ad ogni nodo popolato il tipo di vicinato """
        self.ngh = tuple(neighborhoods)
        self.all_ngh:dict[Nd, Neighborhood] = {}
        self.czk:tuple[str] = self._find_cz_kind() # la lista dei tipi di cittadini presenti in città
       
    def _find_cz_kind(self) -> tuple[str]:
        """ trova i tipi di cittadini presenti in città """
        kd = set() 
        for ng in self.ngh:
            kd.update(tuple(cz.kind for cz in ng.tpr.keys()))
        return tuple(kd)
    
    def __repr__(self) -> str: 
        return f'pop_mix: {len(self.ngh)}, cz_kind = {self.czk}, sn_nodes: {len(self.all_ngh)}'
    
    """ METODI DI GENERAZIONE DEI VICINATI """
         
    def all_equal(self, nodes:Iterable[Nd], idx_ng:int): 
        """ nodes -> i nodi a cui assegnare una tipologia di vicinato 
            idx -> l'indice della tipologia di vicinato 
            tutti i nodi passati in input avranno la stessa composizione """   
        self.all_ngh.update({nd:self.ngh[idx_ng].copy() for nd in nodes})
    
    def all_random(self, nodes:Iterable[Nd]):
        """ nodes -> i nodi a cui assegnare una tipologia di vicinato 
            ad ogni nodo verrà assegnata una tipologia scelta in maniera
            random tra quelle in self.ngg """
        self.all_ngh.update({nd:random.choice(self.ngh).copy() for nd in nodes})
    
    def all_with_distr(self, nodes:Iterable[Nd], prob_dict:dict[int, float]):
       """ nodes -> i nodi a cui assegnare una tipologia di vicinato 
           ad ogni nodo verrà assegnata una tipologia scelta in maniera
           random secondo la distribuzione definita dal dizionario 
           prob_dict, che ha indice (del tipo di cittadino) e probabilità """
           
       n_nodes = len(nodes)
       p_dct = {idx:int(n_nodes*prob_dict[idx]) for idx in prob_dict.keys()}
       rem_nodes = n_nodes - sum(p_dct.values())
       if rem_nodes > 0: # se resta qualcosa lo aggiungiamo al più grosso
           k, _ = sorted([kv for kv in prob_dict.items()], key = lambda x: x[1])[-1]
           p_dct[k] += rem_nodes # eventuali nodi rimanenti li assegnamo al maggiore
       
       self.all_ngh.update({nd:self.ngh[idx_ng].copy() for nd, idx_ng in zip(nodes, iter_prob(p_dct))})    
    
    def __getitem__(self, nd:Nd) -> Neighborhood|None:
        return self.all_ngh.get(nd, None)
    
    def __setitem__(self, nd:Nd, ng:Neighborhood):
        self.all_ngh[nd] = ng
    
    def __iter__(self) -> Generator[tuple[Nd, Neighborhood], None, None]:
        """ restituise, una ad una, le coppie nodo-vicinato """
        for nd, ng in self.all_ngh.items():
            yield nd, ng
    
    @staticmethod
    def gen_node(start_node = (0, 0), nr = 1, nc = 1, *, step_r = 1, step_c = 1) -> Generator[Nd, None, None]:
        """ genera insiemi di nodi contigui (quadranti rettangolari), 
               nr ->  quante righe
                  nc -> quante colonne """
        rs, cs = start_node
        for r in range(rs, rs + nr, step_r):
            for c in range(cs, cs + nc, step_c):
                yield r, c
    
    @staticmethod
    def make_frame(mx:int, my:int, *, layer:int = 0):
       """ mx e my sono il valore massimo delle coordinate x e y 
           dei nodi perimetrai (massimo assoluto), indicizzati da 0"""
       # le chiavi rappresentano i nodi che fanno cambiare la direzione 
       if mx%2 == 0 or my%2 == 0: raise Exception('Estremi errati') # numero nodi pari, quindi ultimo indice dispari
       if layer > mx//2: raise Exception('Layer inesistente')
     
       layer > mx//2
       directions = {
           (layer, layer): move(0, 1), # up
           (layer, my - layer): move(1, 0), # dx
           (mx -layer, my - layer): move(0, -1), # dwn
           (mx-layer, layer ): move(-1, 0)
           } # sx
       node = (layer, layer) 
       direction = directions[node]
       while True:
           yield node
           direction = directions.get(node, direction)
           node = direction(*node)
           if node == (layer, layer): break
    
    @staticmethod
    def make_multi_frame(mx:int, *, my:Optional[int] = None, layers:int|tuple = 0):
        """ mx e my sono il valore massimo delle coordinate x e y 
            dei nodi perimetrai (massimo assoluto) 
            layers sono le cornici da fare """
        if not my: my = mx
        if type(layers) == int: layers = (layers, )
        for lr in layers:
            yield from All_Neighborhood.make_frame(mx, my, layer = lr)
            
class DMatrix:
    """ La classe che gestisce una matrice delle distanze 
           matrice definita come dizionario nodo-nodo: distanza """
    
    def __init__(self, Dm:dict[tuple[Nd, Nd], float]):
        """ Dm è la matrice delle distanze, non è necessario che sia 
            simmetrica. Se è simmetrica basta la parte superiore """
        self.dm = Dm
    
    def __getitem__(self, ns:tuple[Nd, Nd]) -> float:
        """ restituisce la distanza tra due nodi, se collegati """
        n1, n2 = ns
        if n1 == n2: return 0
        try: return self.dm[ns]
        except: return self.dm.get((n2, n1), math.inf) # se non c'è un collegamento allora distanza infinita!!!
    
    def all_nodes(self) -> list[Nd]:
        """ la listsa di tutti i nodi del grafo """
        nds = []
        for n1, n2 in self.dm.keys():
            if n1 not in nds: nds.append(n1)
            if n2 not in nds: nds.append(n2)
        return nds
                
    def _as_df(self) -> Df:
        """ Converte la matrice in un data frame
            per poterla visualizzare graficamente """ 
        nds = self.all_nodes()
        # crea il dizionario che verrà convertito in DF
        # il formato è il seguente n1: {n1:0, n2:5, ..., nn = 7} ... nn:{n1:7, ..., nn:0}
        dct = {str(nd):{str(ot_nd):self[nd, ot_nd] for ot_nd in nds} for nd in nds}
        return pd.DataFrame.from_dict(dct)
    
    def __repr__(self) -> str:
        return repr(self._as_df())
    
class City:
    """ L'intera città popolata """
    
    def __init__(self, env, 
                       n_nodes:int, # numero totale di nodi
                       distance:float, # distanza fissa tra nodi adiacenti
                       the_map: All_Neighborhood, *, 
                       bins:Optional[tuple[Bin]] = None, # bidoni
                       in_out: Optional[tuple[Nd]] = None,
                       calendar:Optional[Calendar] = None): # nodo di input e nodo di output
        
        """ 
        Attributi di base
        non andrebbero mai modificati, ma non ho messo _ per pigrizia @_@ 
        """
        self.env = env
        self.d = distance 
        self.nn = int(n_nodes**0.5) # numero di nodi per riga e per colonna  
        self.map: All_Neighborhood = the_map 
        self.calendar = calendar
        # input e outpt 
        # se non passati in input vengono messi nei due vertici opposti: basso-sx, alto-dx"""
        try: self.n_in, self.n_out = in_out
        except: self.n_in, self.n_out = (0, 0), (self.nn - 1, self.nn - 1)      
        self.gf:Gr = self._gen_graph() # il grafo 2d
        # bins
        # se non vengono definiti ne mettiamo 4 disposti a croce
        if bins: self.bins = bins 
        else: self.bins = Bin(env = env, capacity = random.randint(500, 1000), threshold = None).gen_a_croce(lw = 0, md = self.nn//2, hg = self.nn - 1)         
        # aggiungiamo il colore ai nodi con bin e assegnamo le coordinate cartesiane ai bin
        for B in self.bins:
            self.gf.nodes[B.nd]['color'] = "yellow"
            B.crd = self.gf.nodes[B.nd]['coord'] 
        # creiamo la popolazione di ogni nodo
        # oss. ngh è un oggetto callable che crea la popolazione, 
        # gli passiamo i bin e la distanza dal nodo di riferimento per completare la generazione
        self.ctzs: dict[Nd:tuple[Citizen]] = {nd:ngh(
                                                        env = self.env, 
                                                        center = self.gf.nodes[nd]['coord'], 
                                                        radius = round(self.d/2, 4),
                                                        nd = nd, 
                                                        bins = {b:nx.shortest_path_length(self.gf, source = nd, target = b.nd, weight='distance') for b in self.bins},
                                                        calendar = self.calendar
                                                        )
                                                for nd, ngh in self.map} # i cittadini di ogni quartiere
        # aggiorniamo la popolazione associata ad ogni nodo 
        for nd, cs in self.ctzs.items(): self.gf.nodes[nd]['population'] = len(cs)
        # aggiungiamo gli influencer ad ogni cittadino
        ctr = 0
        tot = 0 
        print('Creazione vicinato')
        for cz in self.iter_citizens(): 
            self._get_influencers(cz)
            ctr += 1
            if ctr == 5000:
                tot += ctr
                print(f'creati {tot} vicinati')
                ctr = 0
        print('Fine creazione vicinato')
        
        """ Attributi Calcolati """
        # numero di cittadini per tipologia 
        # numero di citadini totali
        self.population = {kd:self.n_citizens(flt = partial(lambda k, cz: cz.att.kind == k, kd)) for kd in self.map.czk}
        self.population['global'] = self.n_citizens()
        print('Fine generazione cittadini e vicinato')
        
    def _gen_graph(self) -> Gr:
        """ genera un grafo 2D a griglia
            i nodi vengono generati ed etichettati in questo modo:
               - prima riga bassa (0, 0), (0, 1),...,(0, n)
                   ...
               - ultima riga alta (n, 0), (n, 1),..., (n, n) """ 
        G = nx.grid_2d_graph(self.nn, self.nn)
        for nd, (x, y) in zip(G.nodes(), gen_coord(self.nn, self.nn, self.d)):
            if nd not in (self.n_in, self.n_out): G.nodes[nd]['color'] = "skyblue" # colore di base
            else: G.nodes[nd]['color'] = "green" # input output
            G.nodes[nd]['population'] = 0
            G.nodes[nd]['coord'] = (x, y) # assegnamo coordinate cartesiane ai nodi
        for i, j in G.edges():
            G[i][j]['distance'] = self.d
        return G
    
    def _get_influencers(self, cz:Citizen):
        """ cerca gli influencer entro un raggio predefinito """
        for nd in gen_neighbor(nd = cz.nd, n_layer = cz.att.n_depth): # circondari da considerare
            for other in self.ctzs.get(nd, ()): # altri cittadini nel nodo nd, se il nodo non esiste (i limitrofi dei nodi di bordo potrebbero non esistere), viene restituita una tupla vuota!!!
                if (other is not cz) and ((d:= math.dist(cz.xy, other.xy)) <= cz.rof):
                    cz.infs[other] = d # aggiunge la distanza da ogni citizen influencer
    
    """ Proprietà e metodi di visualizzazione """
    
    def __repr__(self) -> str:
        return f'nodes: {self.n_nodes}, bins: {len(self.bins)}, population: {self.population}'
    
    def show(self, distance:bool = False, population:bool = False, fg_size:tuple[int, int] = (14, 14), nd_size:int = 900):
        """ Mostra il grafo della città con indicazioni sul numero di residenti """
        G = self.gf
        palette = tuple(nx.get_node_attributes(G,'color').values())
        pos = {(x, y): (y, x) for x, y in G.nodes()}  # Disposizione a griglia con (x, y) come coordinate
        plt.figure(figsize = fg_size)
        wl = False if population else True
        nx.draw(G, pos, with_labels = wl, node_color = palette, node_size = nd_size, font_weight = "bold")
        
        if distance: 
            distance_dict  = nx.get_edge_attributes(G,'distance')
            nx.draw_networkx_edge_labels(G, pos, edge_labels = distance_dict)
            
        if population:
            population_dict = nx.get_node_attributes(G, 'population')
            pop_labels = {nd: f'{nd}\n{pop}' for nd, pop in population_dict.items()}
            nx.draw_networkx_labels(G, pos, labels = pop_labels, font_size = 11, font_weight = "bold", font_color = 'black')
        
        plt.title("The city")
        plt.show()
    
    @property
    def n_nodes(self) -> int:
        return self.nn**2
    
    @property
    def l_edge(self) -> float:
        return self.d
    
    
    def gen_dmat(self, show_increase:float = 0)-> DMatrix:
        """ Genera la matrice delle distanze """
        if show_increase:
            dmat = {}
            dn = ((self.nn**2)**2)*show_increase
            perc, done = 0, 0
            for n1, n2 in itertools.product(self.gf.nodes(), repeat = 2):
                dmat[(n1, n2)] = nx.shortest_path_length(self.gf, source = n1, target = n2, weight='distance')
                done += 1
                if done >= dn: 
                    perc += show_increase 
                    print(f'Siamo al {perc:.2%} dei valori')
                    done = 0
            return DMatrix(dmat)
        return DMatrix({(n1, n2): 
                            nx.shortest_path_length(self.gf, source = n1, target = n2, weight='distance') 
                            for n1, n2 in itertools.product(self.gf.nodes(), repeat = 2)})
    
    
    """ Altri Metodi """        
    
    def iter_citizens(self) -> Generator[Citizen, None, None]:
        """ restituisce tutti i cittadini """
        for cs in self.ctzs.values():
            yield from iter(cs)
    
    def n_citizens(self, nodes:Optional[Iterable[Nd]] = None, flt:Callable[[Citizen], bool] = lambda ctz: True) -> int:
        """ numero di cittadini, eventualmente filtrato per quartiere/nodo e altri criteri """
        if nodes is None: nodes = tuple(self.ctzs.keys())
        return sum(sum(1 for ctz in self.ctzs[nd] if flt(ctz)) for nd in nodes)
    
    def bin_coord(self, B:Bin|int = 0) -> tuple[float, float]:
        """ restituisce la coordinata di un bin. Un po' obsoleto dato
            che ho aggiunto la coordinata al bin stesso """
        if not isinstance(B, Bin): B = self.bins[B] 
        nd = B.nd # il nodo
        return self.gf.nodes[nd]['coord']
    
    def node_coord(self, nd:Nd) -> tuple[float, float]:
        """ le coordinate di un nodo """
        return self.gf.nodes[nd]['coord']
    
    def node_pop(self, nd:Nd) -> tuple[float, float]:
        """ la popolazione di un nodo """
        return self.gf.nodes[nd]['population']
    
    def get_cz(self, nd:Nd, idx:int = 0, ctz_type:Optional[str] = None) -> Citizen|None:
        """ uno specifico cittadino (idx) di un certo nodo (nd) 
            o il primo del tipo passato in input"""
        if ctz_type is None:
            try: return self.ctzs[nd][idx]
            except: return None
        else:
            try:
                for cz in self.ctzs[nd]:
                    if cz.att.kind == ctz_type: return cz
            except: return None
        return None
        
"""
ROBA PER DEBUG 


pop_nr:f_pop = normal_pop(500, 50, 50)

#Tre tipologie di cittadino 
C1 = Citizen_type(kind ='eco_friendly', green_awareness = (0.65, 0.75), radius_of_influence = (200, 300), n_depth = 1)
C2 = Citizen_type(kind ='neutral', green_awareness = (0.4, 0.5), radius_of_influence = (200, 300), n_depth = 1)
C3 = Citizen_type(kind ='non_eco', green_awareness = (0.1, 0.25), radius_of_influence = (250, 500), n_depth = 1)

#quattro quartieri tipo 
Ng1 = Neighborhood(population = 100, type_prob = {C2:0.4, C3:0.6})
Ng2 = Neighborhood(population = pop_nr, type_prob = {C1:0.2, C2:0.3, C3:0.5})
Ng3 = Neighborhood(population = 150, type_prob = {C1:1})
Ng4 = Neighborhood(population = pop_nr, type_prob = {C1:0.1, C3:0.9})

#l'insieme dei quartieri abitati 
c_map = All_Neighborhood((Ng1, Ng2, Ng3, Ng4))
n1 = tuple(c_map.gen_node(start_node = (1, 1), nr = 3, nc = 1))
n2 = tuple(c_map.gen_node(start_node = (1, 3), nr = 3, nc = 1))
c_map.all_equal(n1, 0) # tipo 1 sulla colonna n1
c_map.all_random(n2) # tipi a caso sulla colonna n2
c_map[1,2] = Ng2 # tipo Ng2 nel nodo 1-2
c_map[3,2] = Ng4 #tipo Ng4 nel nodo 3-2

env = sp.Environment()  

# I bidoni, messi negli stessi posti in cui verrebero generati in maniera randomica 
B = Bin(env = env, capacity = 1000, threshold = None) 
bins = B.gen_a_croce(lw = 0, md = 2, hg = 4)

# la città 
C = City(env = env, n_nodes = 25, distance = 250, 
         the_map = c_map, bins = bins, in_out = None)

cz = C.get_cz((1, 1))

C.show(True, True)
"""


"""
x = {((0,0), (0, 1)): 10, ((0,1), (1, 0)): 20}
nw = DMatrix(x)
y = nw._as_df()
"""




