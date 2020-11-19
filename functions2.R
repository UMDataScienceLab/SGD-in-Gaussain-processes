#-------------------------------------------------------------#

#--- Functions for case where the kernel is a summation of rbf kernels ---#
#--- Each rbf kernel has a single lengthscale and a single outputscale ---# 

#-------------------------------------------------------------#

calc_dist_sq_mat <- function(X, Y=NULL){
  # Calculate the squared distance matrix of inputs
  # X: matrix or vector
  # Y: matrix or vector
  # Refer to rbfkernel() from "rdetools" package
  
  if (!is.matrix(X)) X <- matrix(X, ncol = 1)
  n <- nrow(X)
  if (is.null(Y)) {
    XtX <- tcrossprod(X)
    XX <- matrix(1, nrow = n) %*% diag(XtX)
    dist_sq_mat <- XX - 2 * XtX + t(XX)
  } 
  else {
    if (!is.matrix(Y)) Y <- matrix(Y, ncol = 1)
    m <- nrow(Y) 
    XX <- matrix(apply(X^2, 1, sum), n, m)
    YY <- matrix(apply(Y^2, 1, sum), n, m, byrow = TRUE)
    XY <- tcrossprod(X, Y)
    dist_sq_mat <- XX - 2 * XY + YY
  }
  
  return(dist_sq_mat)
}

calc_rbf_cov_mat <- function(X, lscale, Y=NULL){
  # Calculate the covariance matrix of an RBF kernel
  # X: input matrix or vector, each row is an observation
  # lscale: lengthscale, all input dimension share a single lengthscale
  
  scaled_X <- X/lscale
  if (is.null(Y)) scaled_Y <- NULL else scaled_Y <- Y/lscale
  dist_sq_mat <- calc_dist_sq_mat(scaled_X, scaled_Y)
  rbf_cov_mat <- exp(-0.5*dist_sq_mat)
  
  return(rbf_cov_mat)
}

calc_matern_cov_mat <- function(X, lscale, p=1, Y=NULL){
  # Calculate the covariance matrix of a matern kernel where nu=p+0.5
  # X: input matrix or vector, each row is an observation
  # lscale: lengthscale, all input dimensions share a single lengthscale
  
  nu <- p+0.5
  scaled_X <- X/lscale
  if (is.null(Y)) scaled_Y <- NULL else scaled_Y <- Y/lscale
  scaled_dist_mat <- calc_dist_sq_mat(scaled_X, scaled_Y) %>% sqrt
  exp_mat <- exp(-sqrt(2*nu)*scaled_dist_mat)
  coef <- gamma(p+1)/gamma(2*p+1)
  poly_mat <- lapply(0:p, function(i) 
    factorial(p+i)/factorial(i)/factorial(p-i)*(sqrt(8*nu)*scaled_dist_mat)^(p-i) 
  ) %>% Reduce("+", .)
  matern_cov_mat <- exp_mat*coef*poly_mat
  
  return(matern_cov_mat)
}

calc_nll <- function(X, y, lscales, oscales, noise_var, kernel="rbf", p=1){
  # Calculate the negative log-likelihood
  # Each lscale corresponds to one oscale
  
  n <- length(y)
  
  if (kernel=="rbf"){
    cov_mat <- mapply(function(a,b) a*calc_rbf_cov_mat(X, b), oscales, lscales, SIMPLIFY = F) %>% 
      Reduce("+", .) + noise_var*diag(n) # covariance matrix with outputscales and noise
  } else if (kernel=="matern"){
    cov_mat <- mapply(function(a,b) a*calc_matern_cov_mat(X, b, p), oscales, lscales, SIMPLIFY = F) %>% 
      Reduce("+", .) + noise_var*diag(n) # covariance matrix with outputscales and noise 
  }
  u_mat <- chol(cov_mat)
  inv_cov_mat <- chol2inv(u_mat)
  log_det <- 2 * sum(log(diag(u_mat)))
  nll <- 0.5*(as.vector(t(y)%*%inv_cov_mat%*%y) + log_det + n*log(2*pi)) / n
  
  return(nll)
}

calc_matern_cov_mat_grad <- function(dist_mat, lscale, p=1){
  
  nu <- p+0.5
  scaled_dist_mat <- dist_mat/lscale
  exp_mat <- exp(-sqrt(2*nu)*scaled_dist_mat)
  coef <- gamma(p+1)/gamma(2*p+1)
  poly_mat <- lapply(0:p, function(i) 
    factorial(p+i)/factorial(i)/factorial(p-i)*(sqrt(8*nu)*scaled_dist_mat)^(p-i) 
  ) %>% Reduce("+", .)
  matern_cov_mat <- exp_mat*coef*poly_mat
  poly_mat_grad <- lapply(0:p, function(i)
    factorial(p+i)/factorial(i)/factorial(p-i)*
      (p-i)*(sqrt(8*nu)*scaled_dist_mat)^(p-i)/(-lscale)
    ) %>% Reduce("+", .)
  
  term1 <- matern_cov_mat*sqrt(2*nu)*scaled_dist_mat/lscale
  term2 <- exp_mat*coef*poly_mat_grad
  matern_cov_mat_grad <- term1 + term2
  
  return(matern_cov_mat_grad)
}


calc_nll_grad <- function(X, y, lscales, oscales, noise_var, kernel="rbf", p=1){
  # Calculate the gradient of the negative log-likelihood function
  # X: matrix or vector
  
  n <- length(y)
  
  if (kernel=="rbf"){
    rbf_cov_mat_list <- lapply(lscales, function(x) calc_rbf_cov_mat(X, x))
    cov_mat <- mapply(function(a,b) a*b, oscales, rbf_cov_mat_list, SIMPLIFY = F) %>% 
      Reduce("+", .) + noise_var*diag(n) # covariance matrix with outputscales and noise
    dist_sq_mat <- calc_dist_sq_mat(X)
    cov_mat_grad_list <- c(mapply(function(o, r, l) o*r*dist_sq_mat/l^3, 
                                  oscales, rbf_cov_mat_list, lscales, SIMPLIFY = F), 
                           rbf_cov_mat_list, 
                           list(diag(n))
    )
  } else if (kernel=="matern"){
    matern_cov_mat_list <- lapply(lscales, function(x) calc_matern_cov_mat(X, x, p))
    cov_mat <- mapply(function(a,b) a*b, oscales, matern_cov_mat_list, SIMPLIFY = F) %>% 
      Reduce("+", .) + noise_var*diag(n) # covariance matrix with outputscales and noise
    dist_mat <- calc_dist_sq_mat(X) %>% sqrt
    cov_mat_grad_list <- c(mapply(function(o, l) o*calc_matern_cov_mat_grad(dist_mat, l, p), 
                                  oscales, lscales, SIMPLIFY = F), 
                           matern_cov_mat_list, 
                           list(diag(n))
    ) 
  }

  inv_cov_mat <- chol2inv(chol(cov_mat))
  inv_cov_mat_y <- as.vector(inv_cov_mat%*%y)
  grad <- sapply(cov_mat_grad_list, function(x) 
    0.5 * (-t(inv_cov_mat_y)%*%x%*%inv_cov_mat_y + sum(diag(inv_cov_mat%*%x))) / n)
  
  return(grad)
}

### Need to be modified to accomodate matern kernel.
calc_pred_mean <- function(X_pred, X, y, lscales, oscales, noise_var){

  n <- length(y)
  pred_cov_mat <- mapply(function(a,b) a*calc_rbf_cov(X_pred, b, X), oscales, lscales, SIMPLIFY = F) %>%
    Reduce("+", .) # with outputscales but without noise variance
  cov_mat <- mapply(function(a,b) a*calc_rbf_cov(X, b), oscales, lscales, SIMPLIFY = F) %>%
    Reduce("+", .) + noise_var*diag(n) # with outputscales and noise variance
  pred_means <- pred_cov_mat%*%solve(cov_mat, y) %>% as.vector

  return(pred_means)
}

### Need to be modified to accomodate matern kernel.
calc_pred_var <- function(X_test, X_train, lscales, oscales, noise_var){
  
  if (!is.matrix(X_train)) X_train <- matrix(X_train, ncol = 1)
  if (!is.matrix(X_test)) X_test <- matrix(X_test, ncol = 1)
  n_train <- nrow(X_train)
  n_test <- nrow(X_test)
  test_cov_mat <- mapply(function(a,b) a*calc_rbf_cov(X_test, b), oscales, lscales, SIMPLIFY = F) %>%
    Reduce("+", .) # with outputscales but without noise variance
  test_train_cov_mat <- mapply(function(a,b) a*calc_rbf_cov(X_test, b, X_train), oscales, lscales, SIMPLIFY = F) %>%
    Reduce("+", .) # with outputscales but without noise variance
  cov_mat <- mapply(function(a,b) a*calc_rbf_cov(X_train, b), oscales, lscales, SIMPLIFY = F) %>%
    Reduce("+", .) + noise_var*diag(n_train) # with outputscales and noise variance
  inv_cov_mat <- chol2inv(chol(cov_mat))
  pred_cov_mat <- test_cov_mat - test_train_cov_mat%*%inv_cov_mat%*%t(test_train_cov_mat)
  
  return(pred_cov_mat)
}
#-------------------------------------------------------------#

# adam <- function(X, y, batch_size, step_size=0.01, max_iter=NULL, n_epochs=25,
#                  n_kers = 1,
#                  init_lscales = NULL, init_oscales = NULL, init_noise_var = NULL,
#                  fix_params = F, param_mins = NULL, beta_1 = 0.9, beta_2 = 0.999, epsilon = 1e-8){
#   # Adam with reparameterization.
#   # X: matrix or vector
#   # params = log(1+exp(thetas)) + param_mins 
#   
#   if (!is.matrix(X)) X <- matrix(X, ncol = 1)
#   param_coefs <- rep(1, (2*n_kers+1))
#   if (fix_params == "lscales") param_coefs[1:n_kers] <- 0
#   if (is.null(param_mins)) param_mins <- c(rep(1e-6, n_kers), rep(1e-6, n_kers), 1e-4)
#   
#   init <- log(2)
#   if (is.null(init_lscales)) init_lscales <- rep(init, n_kers)
#   if (is.null(init_oscales)) init_oscales <- rep(init, n_kers)
#   if (is.null(init_noise_var)) init_noise_var <- init
#   
#   params <- c(init_lscales, init_oscales, init_noise_var)
#   thetas <- log(exp(params - param_mins) - 1)
#   
#   n_samples <- length(y)
#   # Number of batches per epoch
#   n_batches_per_epoch <- n_samples %/% batch_size 
#   if (!is.null(n_epochs)) max_iter <- n_batches_per_epoch*n_epochs
#   param_mat <- matrix(0, nrow = max_iter, ncol = (2*n_kers+1))
#   
#   m <- 0
#   v <- 0
#   alpha <- step_size
#   
#   iter <- 1
#   
#   while (iter <= max_iter){
#     shuffle_indices <- sample(1:n_samples)
#     X_shuffle <- X[shuffle_indices, , drop=F]
#     y_shuffle <- y[shuffle_indices]
#     batch_iter <- 1
#     while ((batch_iter <= n_batches_per_epoch) & (iter <= max_iter)){
#       batch_indices <- ((batch_iter-1)*batch_size+1):(batch_iter*batch_size)
#       batch_X <- X_shuffle[batch_indices, , drop=F]
#       batch_y <- y_shuffle[batch_indices]
#       
#       grad <- calc_nll_grad(batch_X, batch_y, params[1:n_kers], params[(n_kers+1):(2*n_kers)], params[2*n_kers+1])
#       thetas_grad <- param_coefs*grad*(1/(1+exp(-thetas)))
#       t <- iter
#       
#       # Adam
#       g <- thetas_grad
#       m <- beta_1*m + (1-beta_1)*g
#       v <- beta_2*v + (1-beta_2)*g^2
#       m_hat <- m/(1-beta_1^t)
#       v_hat <- v/(1-beta_2^t)
#       thetas <- thetas - alpha*m_hat/(sqrt(v_hat)+epsilon)
#       
#       params <- log(1+exp(thetas)) + param_mins
#       param_mat[iter, ] <- params
#       
#       batch_iter <- batch_iter + 1
#       iter <- iter + 1
#     }
#   }
#   
#   # param_mat has iter-1 rows
#   param_mat <- rbind(c(init_lscales, init_oscales, init_noise_var), param_mat)
#   # param_mat has iter rows
#   adam_updates_df <- cbind(0:(iter-1), param_mat) %>% data.frame
#   col_names <- c("iter", paste("lscale", 1:n_kers, sep=""), paste("oscale", 1:n_kers, sep=""), "noise_var")
#   colnames(adam_updates_df) <- col_names
#   
#   return(adam_updates_df)
# }
#-------------------------------------------------------------#

# adam_nn <- function(X, y, batch_size, step_size=0.01, max_iter=NULL, n_epochs=1,
#                     n_kers = 1,
#                     init_lscales = NULL, init_oscales = NULL, init_noise_var = NULL,
#                     fix_params = F, param_mins = NULL, beta_1 = 0.9, beta_2 = 0.999, epsilon = 1e-8){
#   # Adam with reparameterization.
#   # X: matrix or vector
#   # params = log(1+exp(thetas)) + param_mins 
#   
#   if (!is.matrix(X)) X <- matrix(X, ncol = 1)
#   param_coefs <- rep(1, (2*n_kers+1))
#   if (fix_params == "lscales") param_coefs[1:n_kers] <- 0
#   if (is.null(param_mins)) param_mins <- c(rep(1e-6, n_kers), rep(1e-6, n_kers), 1e-4)
#   
#   init <- log(2)
#   if (is.null(init_lscales)) init_lscales <- rep(init, n_kers)
#   if (is.null(init_oscales)) init_oscales <- rep(init, n_kers)
#   if (is.null(init_noise_var)) init_noise_var <- init
#   
#   params <- c(init_lscales, init_oscales, init_noise_var)
#   thetas <- log(exp(params - param_mins) - 1)
#   
#   n_samples <- length(y)
#   # Number of batches per epoch
#   n_batches_per_epoch <- n_samples %/% batch_size 
#   if (!is.null(n_epochs)) max_iter <- n_batches_per_epoch*n_epochs
#   param_mat <- matrix(0, nrow = max_iter, ncol = (2*n_kers+1))
#   
#   nn_idx_mat <- nn2(X, X, k=batch_size)$nn.idx
#   
#   m <- 0
#   v <- 0
#   alpha <- step_size
#   
#   iter <- 1
#   
#   while (iter <= max_iter){
#     shuffle_indices <- sample(1:n_samples)
#     sample_iter <- 1
#     while ((sample_iter <= n_samples) & (iter <= max_iter)){
#       center_idx <- shuffle_indices[sample_iter]
#       batch_indices <- nn_idx_mat[center_idx,]
#       batch_X <- X[batch_indices, , drop=F]
#       batch_y <- y[batch_indices]
#       
#       grad <- calc_nll_grad(batch_X, batch_y, params[1:n_kers], params[(n_kers+1):(2*n_kers)], params[2*n_kers+1])
#       thetas_grad <- param_coefs*grad*(1/(1+exp(-thetas)))
#       t <- iter
#       
#       # Adam
#       g <- thetas_grad
#       m <- beta_1*m + (1-beta_1)*g
#       v <- beta_2*v + (1-beta_2)*g^2
#       m_hat <- m/(1-beta_1^t)
#       v_hat <- v/(1-beta_2^t)
#       thetas <- thetas - alpha*m_hat/(sqrt(v_hat)+epsilon)
#       
#       params <- log(1+exp(thetas)) + param_mins
#       param_mat[iter, ] <- params
#       
#       sample_iter <- sample_iter + 1
#       iter <- iter + 1
#     }
#   }
#   
#   # param_mat has iter-1 rows
#   param_mat <- rbind(c(init_lscales, init_oscales, init_noise_var), param_mat)
#   # param_mat has iter rows
#   adam_nn_updates_df <- cbind(0:(iter-1), param_mat) %>% data.frame
#   col_names <- c("iter", paste("lscale", 1:n_kers, sep=""), paste("oscale", 1:n_kers, sep=""), "noise_var")
#   colnames(adam_nn_updates_df) <- col_names
#   
#   return(adam_nn_updates_df)
# }
#-------------------------------------------------------------#

##--- Adam.

### Need to be modified to accomodate matern kernel.
adam <- function(X, y, n_kers=1, batch_size=1, step_size=0.01, 
                     n_epochs=1, max_iter=NULL,
                     init_lscales=NULL, init_oscales=NULL, init_noise_var=NULL,
                     fix_lscale=F, fix_oscale=F, fix_noise_var=F,
                     lscale_min=1e-6, oscale_min=1e-6, noise_var_min=1e-4,
                     lscale_sample="unif", oscale_sample="unif", noise_var_sample="unif",
                     beta_1=0.9, beta_2=0.999, epsilon=1e-8){
  # Adam with reparameterization.
  # X: matrix or vector
  # params = log(1+exp(thetas)) + param_mins 
  
  if (!is.matrix(X)) X <- matrix(X, ncol = 1)
  param_coefs <- rep(1, (2*n_kers+1))
  lscale_indices <- 1:n_kers
  oscale_indices <- (n_kers+1):(2*n_kers)
  noise_var_index <- 2*n_kers+1
  if (fix_lscale == T) param_coefs[lscale_indices] <- 0
  if (fix_oscale == T) param_coefs[oscale_indices] <- 0
  if (fix_noise_var == T) param_coefs[noise_var_index] <- 0
  param_mins <- c(rep(lscale_min, n_kers), rep(oscale_min, n_kers), noise_var_min)
  
  init <- log(2)
  if (is.null(init_lscales)) init_lscales <- rep(init, n_kers)
  if (is.null(init_oscales)) init_oscales <- rep(init, n_kers)
  if (is.null(init_noise_var)) init_noise_var <- init
  
  params <- c(init_lscales, init_oscales, init_noise_var)
  thetas <- log(exp(params - param_mins) - 1)
  n_samples <- length(y)
  # Number of batches per epoch
  n_batches_per_epoch <- n_samples %/% batch_size 
  if (!is.null(n_epochs)) max_iter <- n_batches_per_epoch*n_epochs
  param_mat <- matrix(0, nrow = max_iter, ncol = (2*n_kers+1))
  
  sample_schemes <- c(lscale_sample, oscale_sample, noise_var_sample)
  sample_scheme_ids <- sapply(sample_schemes, function(x) if (x=="unif") 1 else 2)
  unif_sample <- F
  nn_sample <- F
  if ("unif" %in% sample_schemes) unif_sample <- T
  if ("nn" %in% sample_schemes){
    nn_sample <- T
    nn_idx_mat <- nn2(X, X, k=batch_size)$nn.idx
  }
  
  m <- 0
  v <- 0
  alpha <- step_size
  iter <- 1
  
  # Mix sampling
  if (unif_sample==T & nn_sample==T){
    while (iter <= max_iter){
      # Uniform minibatch
      unif_batch_indices <- sample(1:n_samples, size=batch_size)
      unif_batch_X <- X[unif_batch_indices, , drop=F]
      unif_batch_y <- y[unif_batch_indices]
      # Nearest neighbor minibatch
      center_idx <- sample(1:n_samples, size=1)
      nn_batch_indices <- nn_idx_mat[center_idx,]
      nn_batch_X <- X[nn_batch_indices, , drop=F]
      nn_batch_y <- y[nn_batch_indices]
      
      unif_grad <- calc_nll_grad(unif_batch_X, unif_batch_y, params[lscale_indices], params[oscale_indices], params[noise_var_index])
      nn_grad <- calc_nll_grad(nn_batch_X, nn_batch_y, params[lscale_indices], params[oscale_indices], params[noise_var_index])
      grad_mat <- rbind(unif_grad, nn_grad)
      grad <- c(grad_mat[sample_scheme_ids[1], lscale_indices], 
                grad_mat[sample_scheme_ids[2], oscale_indices], 
                grad_mat[sample_scheme_ids[3], noise_var_index])
      thetas_grad <- param_coefs*grad*(1/(1+exp(-thetas)))
      
      # Adam
      t <- iter
      g <- thetas_grad
      m <- beta_1*m + (1-beta_1)*g
      v <- beta_2*v + (1-beta_2)*g^2
      m_hat <- m/(1-beta_1^t)
      v_hat <- v/(1-beta_2^t)
      thetas <- thetas - alpha*m_hat/(sqrt(v_hat)+epsilon)
      
      params <- log(1+exp(thetas)) + param_mins
      param_mat[iter, ] <- params
      iter <- iter + 1
    }
  }
  
  # Uniform sampling
  if (unif_sample==T & nn_sample==F){
    while (iter <= max_iter){
      # Uniform minibatch
      unif_batch_indices <- sample(1:n_samples, size=batch_size)
      unif_batch_X <- X[unif_batch_indices, , drop=F]
      unif_batch_y <- y[unif_batch_indices]

      grad <- calc_nll_grad(unif_batch_X, unif_batch_y, params[lscale_indices], params[oscale_indices], params[noise_var_index])
      thetas_grad <- param_coefs*grad*(1/(1+exp(-thetas)))
      
      # Adam
      t <- iter
      g <- thetas_grad
      m <- beta_1*m + (1-beta_1)*g
      v <- beta_2*v + (1-beta_2)*g^2
      m_hat <- m/(1-beta_1^t)
      v_hat <- v/(1-beta_2^t)
      thetas <- thetas - alpha*m_hat/(sqrt(v_hat)+epsilon)
      
      params <- log(1+exp(thetas)) + param_mins
      param_mat[iter, ] <- params
      iter <- iter + 1
    }
  }
  
  # Nearest neighbor sampling
  if (unif_sample==F & nn_sample==T){
    while (iter <= max_iter){
      # Nearest neighbor minibatch
      center_idx <- sample(1:n_samples, size=1)
      nn_batch_indices <- nn_idx_mat[center_idx,]
      nn_batch_X <- X[nn_batch_indices, , drop=F]
      nn_batch_y <- y[nn_batch_indices]
      
      grad <- calc_nll_grad(nn_batch_X, nn_batch_y, params[lscale_indices], params[oscale_indices], params[noise_var_index])
      thetas_grad <- param_coefs*grad*(1/(1+exp(-thetas)))
      
      # Adam
      t <- iter
      g <- thetas_grad
      m <- beta_1*m + (1-beta_1)*g
      v <- beta_2*v + (1-beta_2)*g^2
      m_hat <- m/(1-beta_1^t)
      v_hat <- v/(1-beta_2^t)
      thetas <- thetas - alpha*m_hat/(sqrt(v_hat)+epsilon)
      
      params <- log(1+exp(thetas)) + param_mins
      param_mat[iter, ] <- params
      iter <- iter + 1
    }
  }
  
  # param_mat has iter-1 rows
  param_mat <- rbind(c(init_lscales, init_oscales, init_noise_var), param_mat)
  # param_mat has iter rows
  adam_updates_df <- cbind(0:(iter-1), param_mat) %>% data.frame
  col_names <- c("iter", paste("lscale", 1:n_kers, sep=""), paste("oscale", 1:n_kers, sep=""), "noise_var")
  colnames(adam_updates_df) <- col_names
  
  return(adam_updates_df)
}
#-------------------------------------------------------------#

##--- SGD with diminishing step size.

### Need to be modified to accomodate matern kernel.
sgd <- function(X, y, n_ker=1, batch_size, init_step_size, param_coefs=NULL,
                n_epochs=1, max_iter=NULL,
                init_lscales = NULL, init_oscales = NULL, init_noise_var = NULL,
                fix_lscale=F, fix_oscale=F, fix_noise_var=F,
                lscale_min=1e-6, oscale_min=1e-6, noise_var_min=1e-4,
                lscale_sample="unif", oscale_sample="unif", noise_var_sample="unif"){
  # SGD with reparameterization.
  # X: matrix or vector.
  # params = log(1+exp(thetas)) + min_params
  
  if (!is.matrix(X)) X <- matrix(X, ncol = 1)
  if (is.null(param_coefs)) param_coefs <- rep(1, (2*n_kers+1))
  lscale_indices <- 1:n_kers
  oscale_indices <- (n_kers+1):(2*n_kers)
  noise_var_index <- 2*n_kers+1
  if (fix_lscale == T) param_coefs[lscale_indices] <- 0
  if (fix_oscale == T) param_coefs[oscale_indices] <- 0
  if (fix_noise_var == T) param_coefs[noise_var_index] <- 0
  param_mins <- c(rep(lscale_min, n_kers), rep(oscale_min, n_kers), noise_var_min)
  
  init <- log(2)
  if (is.null(init_lscales)) init_lscales <- rep(init, n_kers)
  if (is.null(init_oscales)) init_oscales <- rep(init, n_kers)
  if (is.null(init_noise_var)) init_noise_var <- init
  
  params <- c(init_lscales, init_oscales, init_noise_var)
  thetas <- log(exp(params - param_mins) - 1)
  n_samples <- length(y)
  # Number of batches per epoch
  n_batches_per_epoch <- n_samples %/% batch_size 
  if (!is.null(n_epochs)) max_iter <- n_batches_per_epoch*n_epochs
  param_mat <- matrix(0, nrow = max_iter, ncol = (2*n_kers+1))
  
  sample_schemes <- c(lscale_sample, oscale_sample, noise_var_sample)
  sample_scheme_ids <- sapply(sample_schemes, function(x) if (x=="unif") 1 else 2)
  unif_sample <- F
  nn_sample <- F
  if ("unif" %in% sample_schemes) unif_sample <- T
  if ("nn" %in% sample_schemes){
    nn_sample <- T
    nn_idx_mat <- nn2(X, X, k=batch_size)$nn.idx 
  }
  
  iter <- 1
  
  # Mix sampling
  if (unif_sample==T & nn_sample==T){
    while (iter <= max_iter){
      # Uniform minibatch
      unif_batch_indices <- sample(1:n_samples, size=batch_size)
      unif_batch_X <- X[unif_batch_indices, , drop=F]
      unif_batch_y <- y[unif_batch_indices]
      # Nearest neighbor minibatch
      center_idx <- sample(1:n_samples, size=1)
      nn_batch_indices <- nn_idx_mat[center_idx,]
      nn_batch_X <- X[nn_batch_indices, , drop=F]
      nn_batch_y <- y[nn_batch_indices]
      
      step_size <- init_step_size / iter
      
      unif_grad <- calc_nll_grad(unif_batch_X, unif_batch_y, params[lscale_indices], params[oscale_indices], params[noise_var_index])
      nn_grad <- calc_nll_grad(nn_batch_X, nn_batch_y, params[lscale_indices], params[oscale_indices], params[noise_var_index])
      grad_mat <- rbind(unif_grad, nn_grad)
      grad <- c(grad_mat[sample_scheme_ids[1], lscale_indices], 
                grad_mat[sample_scheme_ids[2], oscale_indices], 
                grad_mat[sample_scheme_ids[3], noise_var_index])
      
      # thetas_grad <- param_coefs*grad*(1/(1+exp(-thetas)))
      # thetas <- thetas - step_size*thetas_grad
      # params <- log(1+exp(thetas)) + param_mins    
      
      params <- params - param_coefs*step_size*grad
      
      param_mat[iter, ] <- params
      iter <- iter + 1
    }
  }
  
  # Uniform sampling
  if (unif_sample==T & nn_sample==F){
    while (iter <= max_iter){
      # Uniform minibatch
      unif_batch_indices <- sample(1:n_samples, size=batch_size)
      unif_batch_X <- X[unif_batch_indices, , drop=F]
      unif_batch_y <- y[unif_batch_indices]
      
      step_size <- init_step_size / iter
      grad <- calc_nll_grad(unif_batch_X, unif_batch_y, params[lscale_indices], params[oscale_indices], params[noise_var_index])
      
      # thetas_grad <- param_coefs*grad*(1/(1+exp(-thetas)))
      # thetas <- thetas - step_size*thetas_grad
      # params <- log(1+exp(thetas)) + param_mins
      
      params <- params - param_coefs*step_size*grad
      
      param_mat[iter, ] <- params
      iter <- iter + 1
    }
  }
  
  # Nearest neighbor sampling
  if (unif_sample==F & nn_sample==T){
    while (iter <= max_iter){
      # Nearest neighbor minibatch
      center_idx <- sample(1:n_samples, size=1)
      nn_batch_indices <- nn_idx_mat[center_idx,]
      nn_batch_X <- X[nn_batch_indices, , drop=F]
      nn_batch_y <- y[nn_batch_indices]
      
      step_size <- init_step_size / iter
      grad <- calc_nll_grad(nn_batch_X, nn_batch_y, params[lscale_indices], params[oscale_indices], params[noise_var_index])
      
      # thetas_grad <- param_coefs*grad*(1/(1+exp(-thetas)))
      # thetas <- thetas - step_size*thetas_grad
      # params <- log(1+exp(thetas)) + param_mins
      
      params <- params - param_coefs*step_size*grad
      
      param_mat[iter, ] <- params
      iter <- iter + 1
    }
  }
  
  # param_mat has iter-1 rows
  param_mat <- rbind(c(init_lscales, init_oscales, init_noise_var), param_mat)
  # param_mat has iter rows
  sgd_updates_df <- cbind(0:(iter-1), param_mat) %>% data.frame
  col_names <- c("iter", paste("lscale", 1:n_kers, sep=""), paste("oscale", 1:n_kers, sep=""), "noise_var")
  colnames(sgd_updates_df) <- col_names
  
  return(sgd_updates_df)
}
#-------------------------------------------------------------#