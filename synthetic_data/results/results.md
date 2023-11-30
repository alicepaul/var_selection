# Results Documentation

## Model 0 -- Using original States 
Defualt hyperparams: 
- BATCH_SIZE 64 
- TARGET_UPDATE 1
- EPSILON_DECAY .99975 

results 0: p10, 100 episodes </br>
results 1: p50, 100 epsiodes </br>
results 2: p50, 100 epsiodes (TARGET_UPDATE 10) </br>
results 3: p50, 250 episodes (TARGET_UPDATE 10,  EPSILON_DECAY .9999) </br>

## Model 1 -- Updated 34 total states, same model is trained during all results
**Optimal m value is now being set for each batch, ensuring algo. efficiency**
Defualt hyperparams: 
- BATCH_SIZE 64 
- TARGET_UPDATE 1
- EPSILON_DECAY .9999

results 4: p50, 250 episodes </br>
results 5: p50, 500 episodes </br>
results 6: p50, 500 episodes </br>
