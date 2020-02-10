### Keras CNNGWP code for the RStudio interface with Tensorflow backend (https://keras.rstudio.com/). The code performs Bayesian optimization of the hyperparameters of the sparse convolutional neural network described in Waldmann et al. (2020; https://doi.org/10.3389/fgene.2020.00025)

The code reads data from the QTLMAS2010ny012.csv file in the AUTALASSO repository. The y-variable (phenotype) is in the first column and the x-variables (SNPs; coded 0,1,2) are in the following columns (comma separated). The data is partitioned into training (generation 1-4) and test data (generation 5).

In order to run your own data, you need to have the same format as QTLMAS2010ny012.csv, unless you specify alternative options in the read.csv() function. You also need to specify which individuals to assign to training and test data, using the ... index of ytot[...] and Xtrsc[...].

The learning rate in "optimizer_adam(lr = 0.00025)", number of epochs "epochs = 250" and batch size "batch_size = 48" need to be tailored to other data sets. The best way is to set "verbose = 1" in "#Fit model" and monitor the validation MSE.

It is most efficient to run S number of replicates in parallel on a dedicated cluster and to average the hyperparameters over the replicates. The averaged hyperparameters should then be used for ensemble prediction in the cnngwp() function preferably running in parallel. Before running the ensemble prediction, cancel out "#list(Score=-MSEmin,Pred = 0)" and make the code below "#Prediction of yhat based on the best model" executable.
