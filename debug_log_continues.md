# Streaming Factor Trajectory Learning for Dynamic Tensor Decomposition


Idea of continues-mode tensor. 

just set independent GP/SDE/LDSs over each mode seperately

inference is achieved by interation over (CEP over each data-llk + LDS-update)

draft link: https://www.overleaf.com/project/6363a960485a46499baef800
doc link: https://docs.google.com/document/d/1BfxqGF_nqIQ4IeaAEcQJr6foi5Uiu4MUgRESlB7Sfn4/edit

To-do-list:
- finish the data process(Beijing) and data-dict design 
  - done

- tau_update: 
  - done

- try first idea: CEP over all mode -> LDS over all mode
  - almost done
  - the merge_for_test function in LDS: doing!!
  - make it runable
  - make the baseline CEP, SVI
  - all done

- second idea: for-loop over each mode: do CEP->LDS
  - for proposed method, converge faster than first one, final results are similar
  - for standard CEP, performence get worse
  - just use the firsr order now

- Tucker-form 
  - done

- specify the dataset and baseline settings
  - done

- make Tucker-version SVI and CEP as baseline  
  - do when prepare the baselines

- align with streaming project:
  - data process (.npy and .txt)
    - Beijing
    - new data
    - leave to Xin?

  - config arg setting - done
  - log function - done

To-Check-List :
- use standard CEP, store the msg of all data-llk?
  - solved 


Optimize-list :
- 

Observation:
- still sensitive to LDS-paras, DAMPING and dataset
- need multi-epochs to converge
- it seems larger Rank, needs larger ls, var and DAMPING
- for Tucker + Matern23, inits of tucker-core-gamma and post_U is crucial,  rand_init for U, randn_init for U
- smaller DAMPING_tau helps conver fast, but easy to get overfitting
- for tucker case, Matern21 is more robust than Matern23 

- small-rand-init on msg_U_m,msg_gamma_m helps a lot on the convergence and robustness of CEP based model

- rand_init seems always outperformence then randn_init for CEP-tensor model (normalized)

- for Beijing_Air data, it's crucial to use the raw time mode (size: 1421), round it will result in bad results




- optimal setting: 
  - CP: (use CEP + LDS + tau_update in a loop style, use rand_init on U)
    
    R=3 Matern23 + DAMPING: 0.4 + ls:0.1 + var:5 on TEMP_PRES_time_2.5 - test-rmse: 0.29
    
    R=5 Matern21 + DAMPING: 0.4 + ls:0.3 + var:10 on TEMP_PRES_time_2.5 - test-rmse: 0.29


  - Tucker:(use CEP + LDS in a loop style, use rand_init on gamma, rand_init on U)
    
    R=3 Matern21 + DAMPING: 0.6 + DAMPING_tau: 0.8 + DAMPING_gamma: 0.1 + ls:0.1 + var:5 on TEMP_PRES_time_2.5 - test-rmse: 0.33

    R=3 Matern23 + DAMPING: 0.5 + DAMPING_tau: 0.8 + DAMPING_gamma: 0.7 + ls:0.1 + var:5 on TEMP_PRES_time_2.5 - test-rmse: 0.29

    R=5 Matern23 + DAMPING: 0.3 + DAMPING_tau: 0.5 + DAMPING_gamma: 0.7 + ls:0.1 + var:5 on TEMP_PRES_time_2.5 - test-rmse: 0.29


- try other dataset / CONTI-DISCT mixure tensor?