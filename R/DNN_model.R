#' Define and train the DNN-SIM model
#'
#' @param formula an object of class "\link[stats]{formula}" (or one that can be coerced to that class): a symbolic description of the model to be fitted.
#' @param data a data frame.
#' @param num_epochs an integer. The number of complete passes through the training dataset.
#' @param NN_width an integer. The width of the neural network. It must be at least 64.
#' @param NN_depth an integer. The depth of the neural network. It must be at least 1.
#' @param learning_rate a numeric value. The initialized learning rate. Must be a positive value, typically less than or equal to 0.1.
#' @param decayRate a numeric value. The multiplicative factor of learning rate decay. Must be between 0 and 1 (exclusive of 0).
#' @param model the model type. It must be be one of "N-GX-D","SN-GX-D","ST-GX-D","N-GX-B","SN-GX-B","ST-GX-B","N-FX","SN-FX","ST-FX","MED-GX-D","MED-GX-B","T-GX-D","T-GX-PB".
#' @param verbatim TRUE/FALSE.If `verbatim` is `TRUE`, then log information from training the DNN-SIM model will be printed.
#' @param CV TRUE/FALSE. Whether use the cross-validation to measure the prediction accuracy.
#' @param CV_K an integer. The number of folders K-folder cross-validation.
#' @param bootstrap TRUE/FALSE. Whether use the bootstrap method to quantify the uncertainty. The bootstrap option ONLY works for the "ST-GX-D" model.
#' @param bootstrap_B an integer. The number of bootstrap iteration.
#' @param bootstrap_num_epochs an integer. The number of complete passes through the training dataset in the bootstrap procedure.
#' @param U_new TRUE/FALSE. Whether use self defined U for the estimation of single index function, g(U).
#' @param U_min a numeric value. The minimum of the self defined U.
#' @param U_max a numeric value. The maximum of the self defined U.
#' @param random_state an integer. The random seed for initiating the neural network.
#' @param pena_coef A positive numeric value. The penalty coefficient for the "T-GX-PB" model. The penalty term is defined as `pena_coef` multiplied by the squared L2 norm of the Bernstein coefficients.
#'
#' @return A list consisting of the point estimation, g function estimation (optional), cross-validation results (optional) and bootstrap results(optional).
#'
#' @details
#' The DNNSIM model is defined as:
#' \deqn{Y = g(\mathbf{X} \boldsymbol{\beta}) + e.}
#' The residuals \eqn{e} follow a skewed T distribution, skewed normal distribution, or normal distribution. The single index function \eqn{g} is assumed to be a monotonic increasing function.
#'
#' @references
#' \insertRef{liu2022bayesian}{DNNSIM}
#' 
#' \insertRef{liu2025}{DNNSIM}
#' 
#' \insertRef{Liu2026}{DNNSIM}
#'
#' @export
#'
#' @examples
#'
#' \donttest{
#' # check python module dependencies
#' if (reticulate::py_module_available("torch") &
#'     reticulate::py_module_available("numpy") &
#'     reticulate::py_module_available("sklearn") &
#'     reticulate::py_module_available("scipy")) {
#'
#'   # set the random seed
#'   set.seed(100)
#'
#'   # simulate some data
#'   df1 <- data_simulation(n=100,beta=c(1,1,1),w=0.3,
#'                          sigma=0.1,delta=10.0,seed=100)
#'
#'   # the cross-validation and bootstrap takes a long time
#'   DNN_model_output <- DNN_model(y ~ X1 + X2 + X3 - 1,
#'                                 data = df1,
#'                                 model = "ST-GX-D",
#'                                 num_epochs = 5,
#'                                 verbatim = FALSE,
#'                                 CV = TRUE,
#'                                 CV_K = 2,
#'                                 bootstrap = TRUE,
#'                                 bootstrap_B = 2,
#'                                 bootstrap_num_epochs = 5,
#'                                 U_new = TRUE,
#'                                 U_min = -4.0,
#'                                 U_max = 4.0)
#'   print(DNN_model_output)
#' }
#' }
#'
#'
DNN_model <- function(formula,
                      data,
                      model,
                      num_epochs,
                      NN_width = 512,
                      NN_depth = 2,
                      learning_rate = 0.001,
                      decayRate = 0.999,
                      verbatim = TRUE,
                      CV = FALSE,
                      CV_K = 10,
                      bootstrap = FALSE,
                      bootstrap_B = 1000,
                      bootstrap_num_epochs = 100,
                      U_new = FALSE,
                      U_min = -4.0,
                      U_max = 4.0,
                      random_state = 100,
                      pena_coef = 0.1) {
  
  # --- Safety Checks ---
  
  valid_models <- c("N-GX-D", "SN-GX-D", "ST-GX-D", 
                    "N-GX-B", "SN-GX-B", "ST-GX-B", 
                    "N-FX", "SN-FX", "ST-FX", 
                    "MED-GX-D", "MED-GX-B", 
                    "T-GX-D", "T-GX-PB")
  
  if (!(model %in% valid_models)) {
    stop(paste("Input Error: 'model' must be one of:", 
               paste(valid_models, collapse = ", ")))
  }
  
  if (!is.numeric(NN_width) || NN_width < 64) {
    stop("Input Error: NN_width must be an integer and at least 64.")
  }
  
  if (!is.numeric(NN_depth) || NN_depth < 1) {
    stop("Input Error: NN_depth must be an integer and at least 1.")
  }
  
  # Learning Rate checks (Reasonable range: 0 < lr <= 0.1)
  if (!is.numeric(learning_rate) || learning_rate <= 0 || learning_rate > 0.1) {
    stop("Input Error: learning_rate must be a positive numeric value (suggested range: 0 to 0.1).")
  }
  
  # Decay Rate checks (Reasonable range: 0 < decay <= 1.0)
  if (!is.numeric(decayRate) || decayRate <= 0 || decayRate > 1.0) {
    stop("Input Error: decayRate must be between 0 and 1.0 (multiplicative factor).")
  }
  
  # --- Data Preparation ---
  num_epochs <- as.integer(num_epochs)
  bootstrap_num_epochs <- as.integer(bootstrap_num_epochs)
  NN_width <- as.integer(NN_width)
  NN_depth <- as.integer(NN_depth)

  # ensure data is an dataframe
  data <- as.data.frame(data)

  # creation of the design matrix
  X_mat <- stats::model.matrix(object = formula, data = data)

  # extract the column names
  coltxt <- colnames(X_mat)

  # define the response variable
  y <- as.numeric(data[,as.character(formula[[2]])])

  # define the number of columns of the design matrix
  p <- ncol(X_mat)

  # initialize the model that need be to trained
  Model_to_Train <- DNN_Model$Models(p = p,
                                     hidden_size = NN_width,
                                     model = model,
                                     depth = NN_depth,
                                     pena_coef = pena_coef,
                                     seed = as.integer(random_state))

  # train the model
  DNN_train_output <- DNN_Model$Train_Model(model = Model_to_Train,
                                            XT = X_mat,
                                            yT = y,
                                            num_epochs = num_epochs,
                                            lr0 = learning_rate,
                                            decayRate = decayRate,
                                            verbatim = verbatim)

  output <- list(DNN_train_output = DNN_train_output)

  # the prediction of g() function
  if (model %in% c("ST-GX-D",
                   "T-GX-D",
                   "SN-GX-D",
                   "N-GX-D",
                   "MED-GX-D",
                   "ST-GX-B",
                   "T-GX-PB",
                   "SN-GX-B",
                   "N-GX-B",
                   "MED-GX-B")) {
    if (U_new) {
      g_function_estimation <- DNN_Model$Model_prediction_gX(model = Model_to_Train,
                                                             beta_estimation = DNN_train_output$beta_output$cpu()$detach()$numpy(),
                                                             start = U_min,
                                                             stop = U_max,
                                                             num = 100)
    } else {
      g_function_estimation <- DNN_Model$Model_prediction_gX(model = Model_to_Train,
                                                             beta_estimation = DNN_train_output$beta_output$cpu()$detach()$numpy(),
                                                             start = min(DNN_train_output$U_output$cpu()$detach()$numpy()),
                                                             stop = max(DNN_train_output$U_output$cpu()$detach()$numpy()),
                                                             num = 100)
    }

    output[["g_function_estimation"]] <- g_function_estimation
  }

  if (CV) {
    CV_function_output <- DNN_Model$CV_function(XT = X_mat,
                                                yT = y,
                                                model_type = model,
                                                hidden_size = NN_width,
                                                depth = NN_depth,
                                                K = CV_K,
                                                num_epochs = num_epochs,
                                                lr0 = learning_rate,
                                                decayRate = decayRate,
                                                verbatim = verbatim)
    output[["CV_function_output"]] <- CV_function_output
  }

  if (bootstrap & model == "ST-GX-D") {
    ST_GX_D_Bootstrap_output <- DNN_Model$ST_GX_D_Bootstrap(Trained_Model = Model_to_Train,
                                                            DNN_train_output = DNN_train_output,
                                                            XB = X_mat,
                                                            hidden_size = NN_width,
                                                            depth = NN_depth,
                                                            num_epochs = bootstrap_num_epochs,
                                                            lr0 = learning_rate,
                                                            decayRate = decayRate,
                                                            num_bootstrap = bootstrap_B,
                                                            verbatim = verbatim)
    output[["ST_GX_D_Bootstrap_output"]] <- ST_GX_D_Bootstrap_output
  }

  DNN_model_output_names <- names(output$DNN_train_output)
  for (j in DNN_model_output_names) {
    try(output$DNN_train_output[[j]] <- output$DNN_train_output[[j]]$cpu()$detach()$numpy(),
        silent = TRUE)
  }

  return(output)
}
