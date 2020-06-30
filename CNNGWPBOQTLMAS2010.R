
#QTLMAS2010 data
X <- read.csv("QTLMAS2010ny012.csv",header=F,sep=",")
ytot <- (X[,1]-mean(X[,1])) #Mean standardize y
Xtrsc <- scale(X[,2:dim(X)[2]]) #Normalize X
Xtrain <- Xtrsc[1:2326,] #Xtrain
ytrain <- ytot[1:2326] #ytrain
Xtest <- Xtrsc[2327:dim(X)[1],] #Xtest
ytest <- ytot[2327:dim(X)[1]] #ytest

#Load Keras package
library(keras)
#Convert input to Keras format
Xtrain <- array_reshape(Xtrain,c(dim(Xtrain)[1],dim(Xtrain)[2],1))
Xtest <- array_reshape(Xtest,c(dim(Xtest)[1],dim(Xtest)[2],1))

#CNNGWP function
cnngwp<-function(filter,kernel,lambda) {
  
  #Create checkpoint directory
  checkpoint_dir <- "checkpoints"
  unlink(checkpoint_dir, recursive = TRUE)
  dir.create(checkpoint_dir)
  filepath <- file.path(checkpoint_dir, 
                        "weights.{epoch:03d}.hdf5")
  
  #Create checkpoint callback
  cp_callback <- callback_model_checkpoint(
    filepath = filepath,
    monitor = "val_mean_squared_error",
    save_weights_only = TRUE,
    save_best_only = TRUE,
    verbose = 0
  )
  
  #Hyperparameters to optimize
  FLAGS <- flags(
    flag_integer("filter", filter),
    flag_integer("kernel", kernel),
    flag_numeric("lambda", lambda)
  )
  
  #CNNGWP model
  model <- keras_model_sequential()
  model %>%
    layer_conv_1d(filters=FLAGS$filter, kernel_size=FLAGS$kernel, 
                  activation='linear', strides=2L,padding = "same",
                  input_shape=c(dim(Xtrain)[2],1)) %>%
    layer_max_pooling_1d(pool_size=2) %>%
    layer_flatten() %>%
    layer_dense(units=1,activation='linear',
                kernel_regularizer=regularizer_l1(l = FLAGS$lambda))
  
  #Cancel out if model structure should be written out
  #summary(model)
  
  #Compile model
  model %>% compile(
    loss = 'mean_squared_error',
    optimizer = optimizer_adam(lr = 0.00025),
    metrics = c('mse')
  )
  
  #Fit model
  history <- model %>% fit(
    Xtrain, ytrain, 
    epochs = 250,
    verbose = 0,
    batch_size = 48,
    validation_data = list(Xtest,ytest),
    callbacks = list(cp_callback)
  )
  
  #Min MSE from validation (test) data
  #MSEmin <- min(history$metrics$val_mean_squared_error)
  MSEmin <- min(history$metrics$val_mse)
  #Negative MSEmin as test score for Bayesian optimization
  list(Score=-MSEmin,Pred = 0)
  
  #Prediction of yhat based on the best model
  #mods<-list.files(checkpoint_dir)
  #bestmod<-mods[length(mods)]
  #model %>% load_model_weights_hdf5(
  #  file.path(checkpoint_dir, bestmod)
  #)
  #yhat <-predict(model,Xtest)
  #return(yhat)
}


#Bayesian optimization of hyperparameters
#15 random initial points
#25 iterations for optimization using ucb as acquisition function
#Kappa and eps default values
library(rBayesianOptimization)
OPT_Res_BO <- BayesianOptimization(cnngwp,
                                   bounds = list(filter = c(20L,100L),
                                                 kernel = c(10L,50L),
                                                 lambda = c(0.1,1.0)),
                                   init_grid_dt = NULL,
                                   init_points = 15,
                                   n_iter = 25,
                                   acq = "ucb", kappa = 2.576, eps = 0.0,
                                   verbose = TRUE)


# Parallel version that should be used with care since the checkpoint directories can be mixed up. It is better to use 
# a dedicated cluster a make sure to use unique checkpoint directories.

#library(doParallel)
#numCores <- detectCores()
#cl <- makeCluster(numCores)
#clusterExport(cl,c("Xtrain","Xtest","ytrain","ytest"))
#clusterEvalQ(cl,require(keras))
#clusterEvalQ(cl,require(rBayesianOptimization))
#registerDoParallel(cl)
#OPT_Res_BO_Par <- foreach(k = 1:numCores) %dopar% {
#  BayesianOptimization(cnngwp,
#                      bounds = list(filter = c(20L,100L),
#                                    kernel = c(10L,50L),
#                                    lambda = c(0.1,1.0)),
#                      init_grid_dt = NULL,
#                      init_points = 15,
#                      n_iter = 25,
#                      acq = "ucb", kappa = 2.576, eps = 0.0,
#                      verbose = TRUE)
#}
#stopCluster(cl)
