# -*- coding: utf-8 -*-

""" ***************************************************************
FUNZIONI STOCASTICHE D'APPOGGIO 

f_waste_gen --> triangolare per la generazione dei rifiuti

f_influence_sig --> restituisce la probabilità che ci sia una modifica 
              di green awareness (gaw) 
                 a) legato all'infuenza del vicinato o 
                 b) legato all'efficienza del sistema di smaltimento (bidone pieno o vuoto)

f_dist_sig --> effetto disincentivante 

f_incentive_tr --> triangolare che restituisce la percentuale di
             incremento di gaw per effetto di incentivo

f_move --> usa un'uniforme per dire se vanno a piedi o in auto             

*************************************************************** """
import random
import math
from enum import Enum
from typing import Optional, Iterable, NamedTuple, Callable, Any, Generator

def f_waste_gen(low:int, mode:int, high:int) -> float:
    """ materiale tessile generato """
    def inner():
        return random.triangular(low, mode, high)
    inner.__name__ = f'Wg_triangular({low}, {mode}, {high})'
    return inner

def sigmoide_asimmetrica(x:float, alpha:float, beta:float) -> float:
    if x >= 1: return 1
    if x <= 0: return 0
    return x**alpha / (x**alpha + (1 - x)**beta)

class Influence(Enum):
    People = "People"
    Bin_full = "Bin_full"
    Bin_empty = 'Bin_empty'

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
                delta = -abs(delta_dec) if x > 0 else abs(delta_inc)
            case Influence.Bin_full:
                delta = -abs(delta_dec)
                x = 1 - x
            case Influence.Bin_empty:
                delta  = abs(delta_inc)
            case _: raise Exception(""" Tipo d'influenza ignota """)   
            
        rnd = random.random()
        prob = sigmoide_asimmetrica(abs(x), alpha, beta)
        if rnd <= prob: return delta #return prob, delta
        return 0 # prob, 0
   inner.__name__ = f'f_influence_sigmoide(a = {alpha}, b = {beta}, d+ = {delta_inc}, d- = {delta_dec})'
   return inner 

def f_dist_sig(alpha:float, beta:float, max_ds: float, in_km:bool) -> Callable:
    """ 
        serve a modellizzare l'effetto disincentivante legato alla distanza dal bin più vicino
            se la distanza è troppa la green awareness cala...  
                 il risulato, è una coppia di valori (estremo inferiore e superiore)
                 che verranno utilizzati per definire, tramite uniforme, la percentuale
                 di decremento che verrà utilizzata come fattore moltiplicativo 
                 della gaw iniziale """
                           
    if in_km: max_ds *= 1_000
    def inner(ds:float) -> float:
        """ una doppia funzione sigmoide (superiore assimmetrica e inferiore simmetrica) 
               che determina il range di variazione
                    vedi foglio excel allegato. 
                       La penalità è estratta casuamente nel range così generato"""
        if ds <= 0 : return 0
        if in_km: ds *= 1_000
        if ds >= max_ds: return 1
        ds /= max_ds
        estremo_inf =  sigmoide_asimmetrica(ds, alpha = alpha, beta = alpha)
        estremo_sup =  sigmoide_asimmetrica(ds, alpha = alpha, beta = beta)
        if estremo_inf > estremo_sup:
            estremo_sup, estremo_inf = estremo_inf, estremo_sup 
        #return random.uniform(estremo_inf, estremo_sup)
        return random.uniform(estremo_inf, estremo_sup)
       
        inner.__name__ = f'f_doppua_sigmoide(alpha: {alpha}, beta; {beta})'
    return inner

def f_incentive_tr(low:float, mode:float, high:float) -> Callable:
    """ 
    effetto dell'incentivo 
        la percntuale d'incremento che verrà usata come fattore moltiplicativo
        della gaw ad ogni anno"""
    def inner() -> float:
        return random.triangular(low, mode, high)
    inner.__name__ = f'f_incentive_tr({low}, {high})'
    return inner

def f_move(d_max:float, v_feet:float, v_car:float, *, treshold_perc:tuple = (0.5, 1.2)) -> Callable:
    """ tipo e tempo di spostamento, 
            se output booleano è vero, allora va a piedi """
    def inner(km:float)-> tuple[bool, float]:
        l_th, h_tr = treshold_perc
        thr = random.uniform(min(0.1, d_max*l_th), d_max*h_tr) # modifichiamo di volta in volta il treshold
        if km <= thr: return True, round(2*km/v_feet, 4)
        return  False, round(2*km/v_car, 4)
    inner.__name__ = f'f_move(d_max = {d_max})'
    return inner  