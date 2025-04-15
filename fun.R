Trans.fun=function(n.s, x.t, y.t, lam.const, w.fit, family="binomial"){
  n.t=length(y.t)
  w.fit<-w.fit*(abs(w.fit)>=lam.const*sqrt(2*log(p)/n.s))
  myoffset=x.t%*%w.fit
  delta.fit <- as.numeric(glmnet::glmnet(x=x.t,y=y.t, offset=myoffset, lambda=lam.const*sqrt(2*log(p)/n.t), family=family)$beta)
  delta.fit<-delta.fit*(abs(delta.fit)>=lam.const*sqrt(2*log(p)/n.t))
  beta.fit <- w.fit + delta.fit
  beta.fit
}

evaluate_model <- function(w, x.v, y.v, family) {
  y.v=as.vector(y.v)
  # Apply model to the validation set
  pred <- predict(glmnet::glmnet(x.v, y.v, lambda = lam.const), newx = x.v, type = "response")
  if (family == "binomial") {
    # For binary classification, calculate AUROC and AUPRC
    auc <- pROC::roc(y.v, pred)$auc
    prc <- PRROC::pr.curve(scores.class0 = pred, weights.class0 = y.v)$auc.integral
    return(c(AUROC = auc, AUPRC = prc))
  }
  # For other families, you can expand the evaluation logic
  return(NULL)
}