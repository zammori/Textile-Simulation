# -*- coding: utf-8 -*-
"""
Created on Tue May 13 16:46:05 2025

@author: zammo
"""
import random 
from typing import Optional, Callable
from enum import Enum

class Influence(Enum):
    People = "People"
    Bin_full = "Bin_full"
    Bin_empty = 'Bin_empty'
 
def sigmoide_asimmetrica(x:float, alpha:float, beta:float) -> float:
    if x >= 1: return 1
    if x <= 0: return 0
    return x**alpha / (x**alpha + (1 - x)**beta)

#def f_influence_sig(alpha:float = 1.7, beta:float = 8, delta_inc:float = 0.1, delta_dec:float = -0.2) -> Callable:
   
def f_influence_sig(alpha:float, beta:float, delta_inc:float, delta_dec:float, *, inf_type:Influence = Influence.People) -> Callable:
   
   """ Funzione che dà la probabilità che l'agente venca influenzato, 
       con influenza modellizzata usado una sigmoide smussata asimmetrica.
           
           Caso 1) Influence.People, allora si tratta dell'influenza reciproca.
               L'argomento x (input di inner) è la differenza (meglio se percentuale)
               tra la gaw dell'agente e quella media dei suoi vicini. Se x è positivo
               allora l'atente riduce la sua gaw (-delta_dec) e viceversa 
          
           Caso 2) Influence.Bin, allora si tratta del livello di servizio (bidone pieno o vuoto).
                   In questo caso x (input di inner) è 1 - gaw se bin_full (una persona con gaw alta non si demoralizza se bidone pieno)
                   Mentre x è proprio pari alla gaw, se bin_empty """
   def inner(x:float) -> float:
        match inf_type:
            case Influence.People:
                delta = -abs(delta_dec) if x < 0 else abs(delta_inc)
            case Influence.Bin_full:
                delta = -abs(delta_dec)
                x = 1 - x
            case Influence.Bin_empty:
                delta  = abs(delta_inc)
            case _: raise Exception(""" Tipo d'influenza ignota """)   
            
        rnd = random.random()
        prob = sigmoide_asimmetrica(abs(x), alpha, beta)
        if rnd <= prob: return prob, delta
        return prob, 0
   inner.__name__ = f'f_infl_sigmoide(a = {alpha}, b = {beta}, d+ = {delta_inc}, d- = {delta_dec})'
   return inner 

def f_incentive_tr(low:float, mode:float, high:float) -> Callable:
    """ 
    effetto dell'incentivo 
        la percetauel d'incremento che verrà usata come fattore moltiplicativo 
        (1 + x) del gr_aw"""
    def inner() -> float:
        return random.triangular(low, mode, high)
    inner.__name__ = f'f_incn_tr({low}, {high})'
    return inner

f_people = f_influence_sig(1.7, 8, 0.1,-0.2, inf_type = Influence.People)
f_binF = f_influence_sig(1.2, 5, None, -0.2, inf_type = Influence.Bin_full)
f_binE = f_influence_sig(1.2, 5, 0.1,None , inf_type = Influence.Bin_empty)

a = f_people(-0.2) 

# alpha:float = 2, beta:float = 4, max_ds:4000, in_km:bool = True
def f_dist_sig(alpha:float, beta:float, max_ds: float, in_km:bool) -> Callable:
    """ 
        in_km = True se i valori sono passati in km e non in metri 
        effetto disincentivante legato alla distanza dal bin più vicino
            se la distanza è troppa la green awareness cala...  
                 il risulato, similmente a quello di f_incentive 
                        è la percetuale d'incremento che verrà usata come 
                            fattore moltiplicativo  gr_aw  """
    if in_km: max_ds *= 1_000
    def inner(ds:float) -> float:
        """ una doppia funzione sigmoide (superiore assimmetrica e inferiore simmetrica) 
               che determina il range di variazione
                    vedi foglio excel allegato """
        if ds <= 0 : return 0
        if in_km: ds *= 1_000
        if ds >= max_ds: return 1
        ds /= max_ds
        estremo_inf =  sigmoide_asimmetrica(ds, alpha = alpha, beta = alpha)
        estremo_sup =  sigmoide_asimmetrica(ds, alpha = alpha, beta = beta)
        if estremo_inf > estremo_sup:
            estremo_sup, estremo_inf = estremo_inf, estremo_sup 
        #return random.uniform(estremo_inf, estremo_sup)
        return estremo_inf, estremo_sup
       
        inner.__name__ = f'f_doppua_sigmoide(alpha: {alpha}, beta; {beta})'
    return inner


def foo(n_dict:dict = {1:5, 2:3, 3:2}):
    dct = n_dict.copy()
    while True:
        rem = sum(dct.values())
        if not rem: return  
        key = random.choices(list(dct.keys()), weights = tuple(dct[k]/rem for k in dct.keys()))[0]
        yield key
        dct[key] -= 1

def move(dx, dy):
    def inner(x, y): 
        return x + dx, y + dy
    return inner        

def all_random_dist(nodes, prob_dict, ngh):     
   n_nodes = len(nodes)
   p_dct = {idx:int(n_nodes*prob_dict[idx]) for idx in prob_dict.keys()}
   rem_nodes = n_nodes - sum(p_dct.values())
   if rem_nodes > 0: # se resta qualcosa lo aggiungiamo al più grosso
       k, _ = sorted([kv for kv in prob_dict.items()], key = lambda x: x[1])[-1]
       p_dct[k] += rem_nodes # eventuali nodi rimanenti li assegnamo al maggiore
   out_dct = {}
   out_dct.update({nd:ngh[idx_ng] for nd, idx_ng in zip(nodes, foo(p_dct))})    
   return out_dct

def make_frame(mx:int, my:int, *, layer:int = 0):
   """ mx e my sono il valore massimo di coordinata x e y """
   """ le chiavi rappresentano il nodo iniziale che fa prendere una direzione """
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
       
def make_multi_frame(mx:int, my:int, n_layers:int,*, layer:int = 0):
    for i in range(n_layers):
        yield from make_frame(mx, my, layer = layer + i)
        
        
nodes = tuple(make_multi_frame(5, 5, n_layers = 1))
ngh = ('a', 'b', 'c')
pr_dct = {0:0.5, 1:0.3, 2:0.2}
d = all_random_dist(nodes, pr_dct, ngh)



