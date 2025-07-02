# -*- coding: utf-8 -*-
"""
Created on Sat May 31 15:30 2025
@author: zammo

SCENARIO STANDARD - Con gradiente lineare, vedi foglio excel

UNITA' DI GRANDEZZA
tempo in ore
distanze in km
velocità km/h
quantità kg generati al mese 

"""

from typing import Optional, Iterable, NamedTuple, Callable, Any, Generator
from enum import Enum
import math
import random
import matplotlib.pyplot as plt
import pandas as pd
import simpy as sp
import pandas as pd


from Utilities_For_Simulation import *
from Time_Calendar import Calendar
from People import C_Trip, Citizen_type,Citizen
from Map import normal_pop, Neighborhood, All_Neighborhood, DMatrix, City
from Bins import Bin
from Vehicles import Truck
from Sim_Manager import Manager, Calendar, flat_dct, make_df


"*************************************************************************** """
# GENERAZIONE RIFIUTI
f_gen_waste = f_waste_gen(0.75, 1, 1.25) # kg generati nei singoli mesi di bassa stagionalità, distrib triangolare
s_factor = 2.5 # fattore di stagionalità
seasonal_months = (5, 11)
kg_min = 1.5  # quantità minima da conferire

# EFFETTO DISINCENTIVANTE DISTANZA
f_dist = f_dist_sig(alpha = 1.6, beta = 3.2, max_ds = 2.5, in_km = True) # trascurabile sino a 200 metri, gaw si annulla con distanza superiore a 2500

# A PIEDI O IN AUTO 
v_feet = 1.5 #km/h
v_car = 30 #km/h
max_dist = 0.5 #km
threshold = (0.5, 1.2) # fattori moltiplicativi di max_dist
f_mv = f_move(d_max = max_dist, v_feet = v_feet, v_car = v_car, treshold_perc = threshold)

"""****************************************************************************
               DISTRIBUZIONI DIFFERENTI PER TIPOLOGIA DI  CITTADINO
*************************************************************************** """
#INCENTIVI 
# recepiti da ogni cittadino: istante_accensione_incentivo: incentivo
# per ora supponiamo di essere accattivanti per i non_eco e 
un_anno = 365*24
incentive_times = [t*365*24 for t in range(5)] # 0, 8760, 17520, ecc. inizio anno 1, inizio anno 2,...
f_inc_low = f_incentive_tr(0.1,0.15,0.2)
f_inc_std = f_incentive_tr(0.2,0.3,0.4)
f_inc_high = f_incentive_tr(0.3,0.4,0.5)

INC_EC = None # non vengono mai influenzati
INC_NT = {incentive_times[1]: f_inc_std, incentive_times[3]: f_inc_std}  # {incentive_times[1]: f_inc_low, incentive_times[3]: f_inc_low} # vengono influenzati 2 volte 
INC_NE = None # {t:inc for t, inc in zip(incentive_times, (f_inc_high,)*5)} # vengono influenzati sempre

# BIDONE VUOTO (EMPTY)
f_EB_EC = f_influence_sig(alpha = 2.5, beta = 5, delta_inc = 0.015, delta_dec = None , inf_type = Influence.Bin_empty)
f_EB_NT = f_influence_sig(alpha = 2.5, beta = 5, delta_inc = 0.015, delta_dec = None , inf_type = Influence.Bin_empty)
f_EB_NE = f_influence_sig(alpha = 2.5, beta = 5, delta_inc = 0.015, delta_dec = None , inf_type = Influence.Bin_empty)

# BIDONE PIENO (FULL)
f_FB_EC = f_influence_sig(alpha = 2.5, beta = 5, delta_inc = None, delta_dec = -0.15 , inf_type = Influence.Bin_full)
f_FB_NT = f_influence_sig(alpha = 2.5, beta = 5, delta_inc = None, delta_dec = -0.15, inf_type = Influence.Bin_full)
f_FB_NE = f_influence_sig(alpha = 2.5, beta = 5, delta_inc = None, delta_dec = -0.15 , inf_type = Influence.Bin_full)


# PASSAPAROLA - VICINATO
roi = (0.25, 0.35) # raggio influenza3
n_anelli = 2 # anelli di vicinat

# VICINATO 
f_PP_EC = f_influence_sig(alpha = 2.5, beta = 5, delta_inc = 0.2, delta_dec = -0.05, inf_type = Influence.People)
f_PP_NT = f_influence_sig(alpha = 2.5, beta = 5, delta_inc = 0.15, delta_dec = -0.3, inf_type = Influence.People)
f_PP_NE = f_influence_sig(alpha = 2.5, beta = 5, delta_inc = 0.1, delta_dec = -0.15, inf_type = Influence.People)

"""*************************************************
TRE CITTADINI
*************************************************"""
   
eco = Citizen_type(kind ='eco', s_factor = s_factor, m_season = seasonal_months,  min_waste = kg_min,
                   green_awareness = (0.7, 0.9), radius_of_influence = roi, n_depth = n_anelli,
                   gen_waste = f_gen_waste, 
                   green_inc = f_EB_EC, 
                   green_dec = f_FB_EC, 
                   f_influence = f_PP_EC, 
                   f_incentives = INC_EC, 
                   f_dist_penalty = f_dist, 
                   f_move = f_mv) 

neutral = Citizen_type(kind ='neutral', s_factor = s_factor, m_season = seasonal_months, min_waste = kg_min,
                   green_awareness = (0.4, 0.6), radius_of_influence = roi, n_depth = n_anelli,
                   gen_waste = f_gen_waste, 
                   green_inc = f_EB_NT, 
                   green_dec = f_FB_NT, 
                   f_influence = f_PP_NT, 
                   f_incentives = INC_NT,
                   f_dist_penalty = f_dist, 
                   f_move = f_mv) 

non_eco = Citizen_type(kind ='non_eco', s_factor = s_factor, m_season = seasonal_months, min_waste = kg_min,
                   green_awareness = (0.15, 0.35), radius_of_influence = roi, n_depth = n_anelli,
                   gen_waste = f_gen_waste, 
                   green_inc = f_EB_NE, 
                   green_dec = f_FB_NE, 
                   f_influence = f_PP_NE, 
                   f_incentives = INC_NE, 
                   f_dist_penalty = f_dist, 
                   f_move = f_mv) 

"""*************************************************
ISOLATI
*************************************************"""
# POPOLAZIONE PER ISOLATO
pop_nr = normal_pop(mu = 80, sigma = 8, min_pop = 50) # crea popolazione normale, ma si potrebbe passare anche un valore fisso!                         

# TIPI DI ISOLATI
Very_Green = Neighborhood(population = pop_nr, type_prob = {eco:0.75, neutral:0.25}) 
Green = Neighborhood(population = pop_nr, type_prob = {eco:0.375, neutral:0.5, non_eco:0.125}) 
Neutral = Neighborhood(population = pop_nr, type_prob = {eco:0.25, neutral:0.5, non_eco:0.25}) 
Low_Green = Neighborhood(population = pop_nr, type_prob = {eco:0.125, neutral:0.5, non_eco:0.375})
Non_Green = Neighborhood(population = pop_nr, type_prob = {neutral:0.25, non_eco:0.75})

"""*************************************************
LA MAPPA DELLA CITTA'
DIVISA PER ANELLI
*************************************************"""
n_nodi = 18 # numero massimo di nodi per lato 
distanza_nodi = 0.15
mx = n_nodi - 1

The_Map = All_Neighborhood((Very_Green, Green, Neutral, Low_Green, Non_Green))

Estrema_Periferia = tuple(The_Map.make_multi_frame(mx = mx, layers = 0)) # anello estremo (primo)
Periferia = tuple(The_Map.make_multi_frame(mx = mx, layers = (1, 2))) # secondo, terzo                          
Zona_Residenziale = tuple(The_Map.make_multi_frame(mx = mx, layers = (3, 4, 5))) # tre anelli centrali
Centro = tuple(The_Map.make_multi_frame(mx = mx, layers = (6, 7))) # terzultimo e penultimo anello  
Centro_Storico = tuple(The_Map.make_multi_frame(mx = mx, layers = 8))# primo anello  (anello centrale)
                          
The_Map.all_equal(Estrema_Periferia, 4) # tipo Non_Green 
The_Map.all_equal(Periferia, 3) # tipo Low_Green
The_Map.all_equal(Zona_Residenziale, 2) # tipo Neutral
The_Map.all_equal(Centro, 1) # tipo Green
The_Map.all_equal(Centro_Storico, 0) # tipo Very_Green                       
   
"""*************************************************
BIDONI E
MEZZI DI RACCOLTA
*************************************************"""
b_cap = 500 #kg
b_tr =  480
b_ltr = 50 # kg treshold basso, quello per cui il bidone è considerato vuoto 
b_nr = 16

b_coord_est = ((2,2), (2,6), (2,11), (2,15),  (6,15), (11,15), 
               (15, 15), (15, 11), (15, 6), (15, 2), (11, 2), (6,2))

b_coord_int = ((6,6), (6,11), (11,11), (11,6))

b_all_coord = b_coord_est + b_coord_int

start_point = (0, 0)
end_point = (6, 0)

# van che va ai bidoni esterni
big_van_cap = 5000 #kg
big_van_tr =  4800 #kg 
velocità = 25 #km/h
tc_bidone = lambda kg: 0.02 + 0.00016*kg # tempo di carico del bidone, fisso + proporzionale a kg, se bidone pieno 6 minuti
tsc_isola = lambda kg: 1 # tempo fisso di un'ora
# van che va ai bidoni interni
sm_van_cap = 2500 #kg
sm_van_tr = 2300 #kg

intertempo_giro = 14*24 # 14 giorni, 2 settimane


"""*************************************************
SIMULAZIONE 
TEMPI IN ORE, DISTANZE IN KM
QUANTITA' IN KG
*************************************************"""
seeds = (3, 5, 7, 9, 11) #i seed che verranno utilizzati
un_giorno = 24 # 24 ore
giorni_anno = 365
durata_in_anni = 4 # 4 anni
giorni_mese = (31, 28, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31)
intertempo_conferimento = 30*un_giorno # intertempo medio di conferimento
intertempo_agg_gaw = tuple(g*un_giorno for g in giorni_mese) # si può passare anche un numero
distance_matrix = None # questa non cambia, per cui la teniamo buona

DFs = [None, None, None, None, None]

for sd in seeds: # i seed che verranno utilizzati
    s = f'Inizio simulazione con seme {sd}'
    print()
    print('*'*30)
    print(s)
    print()
    random.seed(sd)
    env = sp.Environment()  
    calendar = Calendar(env) # il calendario
    
    # BIDONI
    Std_bin = Bin(env = env, capacity = b_cap, threshold = b_tr, low_treshold = b_ltr, calendar = calendar, full_percent = 0.95)
    bins = Std_bin.gen_n_at(b_all_coord)
    
    # CITTA' E MATRICE DISTANZE
    The_City = City(env = env, n_nodes = n_nodi**2, distance = distanza_nodi, 
                        the_map = The_Map, bins = bins, in_out = (start_point,end_point), 
                        calendar = calendar)
    
    # The_City.show(True, True) # ... per vedere la mappa 
       
    if not distance_matrix: 
        print('Creazione matrice delle distanze, verrà fatto solo una volta')
        distance_matrix = The_City.gen_dmat(show_increase = 0.1)
        print('Fine creazione matrice distanze')
    
    # CAMIONCINI
    trucks = [
                Truck(env = env, the_map = distance_matrix, capacity = big_van_cap, threshold = big_van_tr, 
                      velocity = velocità, n_in = start_point, n_out = end_point, 
                      bins = tuple(b for b in bins if b.nd in b_coord_est),
                      t_load = tc_bidone, t_unload = tsc_isola),
                
                Truck(env = env, the_map = distance_matrix, capacity = sm_van_cap, threshold = sm_van_tr, 
                      velocity = velocità, n_in = start_point, n_out = end_point,
                      bins = tuple(b for b in bins if b.nd in b_coord_int),
                      t_load = tc_bidone, t_unload = tsc_isola),
                ]
    
    Mg = Manager(env = env, the_city = The_City, trucks = trucks, ctz_types = ('eco', 'neutral', 'non_eco'), all_ctz = 'tutti', calendar = calendar)
    
    # I processi che gestisce
    env.process(Mg.calendar.make_calendar(show = True))
    Mg.gen_waste_single_citizen(mean_time = intertempo_conferimento, rnd_bin = True)
    env.process(Mg.update_green_awareness(dt =intertempo_agg_gaw , min_dist = 0.05, wgh = True))
    env.process(Mg.apply_incentives(incentive_times[1:]))
    for tr in Mg.tr_bin.keys():
        env.process(Mg.schedule_trucks(tr, dt = intertempo_giro ))
    print("Inizio effettivo della simulazione")
    
    env.run(until = un_giorno*giorni_anno*durata_in_anni + 5)
    print()
    s = f'Fine simulazione con seme {sd}'
    print(s)
    
    """ Creazione Data_Frames """
    # il dizionario della gaw cumulata nel tempo (per nodo e per tipo di popolazione) viene creato automaticamente
    # i dizionari delle statistiche (per nodo e tipo di cittadini) vanno invece creati  
    Mg.ctz_stat.create_ctzs_stat(Mg.city.iter_citizens()) # stastistiche di base
    Mg.ctz_stat.create_ctzs_stat(Mg.city.iter_citizens(), delta_gaw = True) # variazione di gaw raggruppata per causale
    
    # creiamo i valori cumulati su tutta la città, e sui due anelli (bisogna passare i nodi su cui aggregare, ora ci limitiamo alla città)
    Av_gaw = Mg.ctz_stat.average_gaw(nodes = {'Città':tuple(Mg.ctz_stat.nd_p.keys()), 'Es_periferia':Estrema_Periferia,'Periferia':Periferia, 
                                              'Z_Residenziale' :Zona_Residenziale, 'Centro': Centro, 'C_storico': Centro_Storico}, 
                                                 cz_kinds = ('eco', 'neutral', 'non_eco', 'tutti'))
    
    Av_stat_gen = Mg.ctz_stat.average_ctz_stat(nodes = {'Città':tuple(Mg.ctz_stat.nd_p.keys()), 'Es_periferia':Estrema_Periferia,'Periferia':Periferia, 
                                              'Z_Residenziale' :Zona_Residenziale, 'Centro': Centro, 'C_storico': Centro_Storico}, 
                                                 cz_kinds = ('eco', 'neutral', 'non_eco', 'tutti'),
                                                     Cumulative = True)
    
    Av_stat_dgaw = Mg.ctz_stat.average_ctz_stat(nodes = {'Città':tuple(Mg.ctz_stat.nd_p.keys()), 'Es_periferia':Estrema_Periferia,'Periferia':Periferia, 
                                              'Z_Residenziale' :Zona_Residenziale, 'Centro': Centro, 'C_storico': Centro_Storico}, 
                                                 cz_kinds = ('eco', 'neutral', 'non_eco', 'tutti'),
                                                     Cumulative = False, dgaw = True) 
    
    # i dizionari delle statistiche di bins e track sono creati automaticame te 
    new_df = [
        Mg.ctz_stat.crea_data_frame(Av_gaw, seed = sd, idx_lbl = 0), # 0, 1, 2 definscono il tipo di statistica da creare
        Mg.ctz_stat.crea_data_frame(Av_stat_gen, seed = sd, idx_lbl = 1),
        Mg.ctz_stat.crea_data_frame(Av_stat_dgaw, seed = sd, idx_lbl = 2),   
        Mg.tb_stat.crea_data_frame(seed = sd, ctz = None),
        Mg.tb_stat.crea_data_frame(seed = sd, ctz = Mg.city.iter_citizens())]
    
    for i in range(len(DFs)):
        df_old = DFs[i]
        df_new = new_df[i]
        if df_old is None: DFs[i] = df_new
        else:  DFs[i] = pd.concat([df_old, df_new], ignore_index=True)

""" 
Salviamo i dati su Excel
"""

Excel_Files = {
                '_01 Citizens_BaseCase_SmB.xlsx':
                        [(DFs[0], 'Time_Series_Gaw'), (DFs[1], 'Block_Stats'), (DFs[2], 'Gaw_Components')], 
                '_01 TruckBins_BaseCase_SmB.xlsx':
                        [(DFs[3],'Truck_Stats'), (DFs[4], 'Bin_Stats')]
                }
    
for ex_file, df_sheets in Excel_Files.items():
    with pd.ExcelWriter(ex_file, engine='openpyxl') as writer:
        for df, sh in df_sheets:
            df.to_excel(writer, sheet_name = sh, index=False)
        

""" Giusto per completezza, a fine simulazione mostriamo il grafico gaw finale, 
    e l'andamento nel primo magazzino """       

Stcz = Mg.ctz_stat
Stcz.plot_gaw()
Stbn = Mg.tb_stat
Stbn.show_bin_trend(0)



















