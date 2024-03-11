# Results Documentation
## Training

### Round 0 -- Using original States 
Defualt hyperparams: 
- BATCH_SIZE 64 
- TARGET_UPDATE 1

results 0: p10, 100 episodes

results 1: p50, 100 epsiodes

results 2: p50, 100 epsiodes (TARGET_UPDATE 10)

results 3: p50, 250 episodes (TARGET_UPDATE 10,  EPSILON_DECAY .9999)

### Round 1 -- Updated 34 total states, same model is trained during all results
**Optimal m value is now being set for each batch, ensuring algo. efficiency**
Defualt hyperparams: 
- 34 -> 32 -> 16 -> 1
- BATCH_SIZE 64 
- TARGET_UPDATE 1

results 4: p50, 250 episodes 

results 5: p50, 500 episodes

results 6: p50, 500 episodes

results 7: p50, 500 episodes 
- Comparing against randomly selected algorithm


### Round 2
Defualt hyperparams: 
- 34 -> 64 -> 16 -> 1
- BATCH_SIZE 64 
- TARGET_UPDATE 1

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


### Round 3 -- Using updated Retrobranching Algorithm
- 34 -> 64 -> 16 -> 1
- BATCH_SIZE 64 
- TARGET_UPDATE 1

results 13: p10, Offline Learning ~200 Episodes
- Comparing Offline Max Frac Branch and Strong Branching Experieneces
results 14: p50, Offline Learning ~100 Episodes
- Comparing Offline Max Frac Branch and Strong Branching Experieneces

### Round 4 -- Using Continuous Rewards
- 34 -> 64 -> 16 -> 1
- BATCH_SIZE 64
- TARGET_UPDATE 1

results 15: p10, 100 Episodes
results 16: p50, 500 Episodes
results 17: p50, 500 Episodes 

### Round 5 -- Only Adding Sample of 128 State Pairs to Memory
- 34 -> 64 -> 16 -> 1
- BATCH_SIZE 64
- TARGET_UPDATE 1
- 16 Iterations of Learning after each episode

results 18: p10, 100 Episodes
results 19: p50, 250 Episodes
results 20: p50, 100 Episodes (8 iters)

### Round 5 -- Final Model Training
- 34 -> 64 -> 16 -> 1
- BATCH_SIZE 64
- TARGET_UPDATE 1
- GAMMA 0.8
- 8 Iterations of Learning after each episode

results 21: p10, 100 Episodes
results 22: p50, 100 Episodes
results 23: p100, 87 Episodes
- Training cut short, model training post episode limit saw a decrease in performance
results 24: Mixed Settings, 91 Episodes
- Training cut short, model training post episode limit saw a decrease in performance

## Testing
- Comparing P10, P50, P100 and Mixed Models and Max Fraction Branch and Strong Branching 

### Results
p10_testing: p10, 25 Episodes
p50_testing: p50, 25 Episodes
p100_testing: p100, 25 Episodes