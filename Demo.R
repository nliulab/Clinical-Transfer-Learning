# Demo for reproducing Trans-Lasso Algorithm
# Load the required custom function (assuming it's defined in 'fun.R')
source("fun.R")

#####
#####
##### Step 0. Load simulated data
dat.all <- readRDS("data.sim.rds")

# Load source data (optional if you don't have access to source data)
dat.s <- dat.all$dat.s # Source data
dat.t <- dat.all$dat.t # Target data (always required for refinement)
dat.v <- dat.all$dat.v # Validation data (needed for evaluation)

# Prepare data matrices for training (target data) and source data
x.t <- data.matrix(dat.t[,-1])  # Target data features
y.t <- dat.t[, 1]  # Target data response
x.s <- data.matrix(dat.s[,-1])  # Source data features
y.s <- dat.s[, 1]  # Source data response
x.v <- data.matrix(dat.v[,-1])  # Validation data features
y.v <- dat.v[, 1]  # Validation data response

# Number of predictors and sample size for source data
p <- ncol(x.s)
n.s <- nrow(x.s)

#####
#####
##### Step 1. Get source model parameters
family <- "binomial"  # Model type for binary classification

# Fit the source model using cross-validation to select the best lambda
cv.init <- glmnet::cv.glmnet(x = x.s, y = y.s, nfolds = 8, 
                             lambda = seq(1, 0.1, length.out = 10) * sqrt(2 * log(p) / n.s), 
                             family = family)

# Get the lambda value that minimizes cross-validation error
lam.const <- cv.init$lambda.min / sqrt(2 * log(p) / n.s)

# Fit the source model with the optimal lambda
w.fit <- as.numeric(glmnet::glmnet(x.s, y.s, lambda = lam.const * sqrt(2 * log(p) / n.s), family = family)$beta)

#####
#####
##### Step 2. Conduct Transfer Learning (TL) using target data
w.tl <- Trans.fun(n.s = n.s, x.t = x.t, y.t = y.t, lam.const = lam.const, w.fit = w.fit, family = family)

# Check the difference between source model and TL model coefficients
cat("Source model coefficients:\n")
print(w.fit)
cat("TL model coefficients:\n")
print(w.tl)

#####
#####
##### Additional Evaluation (Self-defined): Compare models using the validation set
# Evaluate both models
cat("Evaluation for Source Model:\n")
eval.source <- evaluate_model(w.fit, x.v, y.v, family)
cat("AUROC and AUPRC for Source Model:", eval.source, "\n")

cat("Evaluation for TL Model:\n")
eval.tl <- evaluate_model(w.tl, x.v, y.v, family)
cat("AUROC and AUPRC for TL Model:", eval.tl, "\n")