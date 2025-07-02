Run_Sim_BaseLine executes 10 simulation runs, each simulating 4 years of operation within an urban network modeled as a square grid composed of 18 × 18 = 324 equally spaced nodes, with a 150-meter distance between neighboring nodes.
Each node hosts an average of 80 citizens. The population is spatially distributed based on environmental awareness: citizens with stronger ecological attitudes are concentrated in the city center, while less environmentally conscious individuals are predominantly located in the outskirts.
The waste collection infrastructure includes 16 bins arranged in two concentric rings and serviced by two collection trucks, each operating on a fixed bi-monthly schedule. While the system is dimensioned to handle the average waste generation, it struggles to manage excess waste during high-seasonal periods.

Run_Sim_Optimistic_Case operates on the same urban network, but introduces incentive policies and replaces standard bins with smart bins. These smart bins are capable of triggering a truck dispatch as soon as a predefined threshold level is reached. This configuration is intended to enhance responsiveness and improve overall service efficiency.

Run_Sim_Pessimistic_Case also uses the same network topology, but reduces the collection frequency by half. As a result, the system becomes severely underdimensioned in relation to the urban waste load, leading to frequent service failures and uncollected waste accumulation.

