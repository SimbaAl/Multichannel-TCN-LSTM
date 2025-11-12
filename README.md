This is the code for Simulating, Preprocessing Training and evaluating a TCN and LSTM on a multichannel data. 

1. First the script IDX_Generation generation the training and testing indexes that are used to generate the training and testing frames.
2. gen_mixed_channel_training.m generates the multi-channel, mixed snr data for the TCN and High SNR multichannel data for the LSTM 
3. gen_mixed_channel_testing.m generates the testing simulation files for each channel model at each SNR (0 to 40). this is differentiated from training by the indices from 1. 
4. build_TCN_dataset_mixedchannel.m Preprocess the training simulation data for NN by interleaving along OFDM symbol dimension.
5. build_TCN_dataset_testing_individual.m Preprocess the testing data for the TCN
6. LSTM_Datasets_Generation.n Preprocess the training data for the LSTM (stacking along subcarrier dimension)
7. build_lstm_data.m Preprocess the testing data for the LSTM. 
8. process_TCN_MixedChannel_result_LEGACY.m Process the TCN estimates, calculate BER and NMSE.
9. process_lstm_results_multichannel.m Process the LSTM estimates, calculate BER and NMSE.
10. resdialcnn_optuna_tuned.py This iis the TCN model script to train and test the TCN model 

