# Results Documentation

## Round 0 -- Using original States 
Defualt hyperparams: 
- BATCH_SIZE 64 
- TARGET_UPDATE 1
- EPSILON_DECAY .99975 

results 0: p10, 100 episodes

results 1: p50, 100 epsiodes

results 2: p50, 100 epsiodes (TARGET_UPDATE 10)

results 3: p50, 250 episodes (TARGET_UPDATE 10,  EPSILON_DECAY .9999)

## Round 1 -- Updated 34 total states, same model is trained during all results
**Optimal m value is now being set for each batch, ensuring algo. efficiency**
Defualt hyperparams: 
- 34 -> 32 -> 16 -> 1
- BATCH_SIZE 64 
- TARGET_UPDATE 1
- EPSILON_DECAY .9999

results 4: p50, 250 episodes 

results 5: p50, 500 episodes

results 6: p50, 500 episodes

results 7: p50, 500 episodes 
- Comparing against randomly selected algorithm


## Round 2
Defualt hyperparams: 
- 34 -> 64 -> 16 -> 1
- BATCH_SIZE 64 
- TARGET_UPDATE 1
- EPSILON_DECAY .9999

results 8: p50, 500 episodes
- Larger Network

results 9: p10, 250 episodes 
- Eliminated Relu on last hidden layer of DQN, was leading to the dying relu problem
- Updated to use leaky_relu, above model was saved as model_pn_1.pt 

results 10: p50, 500 episodes 

results 11: p50, 500 episodes 
- Recreate previous

results 12: p10, 500 episodes 
- Comparing to Strong Branching 


## Round 3 -- Using updated Retrobranching Algorithm
- 34 -> 64 -> 16 -> 1
- BATCH_SIZE 64 
- TARGET_UPDATE 1
- EPSILON_DECAY .9999

results 13: p10, Offline Learning ~200 Episodes
- Comparing Offline Max Frac Branch and Strong Branching Experieneces