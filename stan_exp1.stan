// state-space modeling
// common local levels + difference according to conditions

data{
  int N;
  int N_time;
  real y[N];
  int cond[N];
  int times[N];
  int upperBound;
  int N_id;
  int id[N];
}

parameters{
  real<lower=-1, upper=1> mu[N_time, N_id];
  real<lower=0> s_mu;
  real<lower=0> s_Y;
  
  real<lower=-1, upper=1> diff[2];
  real<lower=1, upper=upperBound> cp1;
}

transformed parameters{
  vector[N] y_true;

  for (i in 1:N){
    if (cond[i] > 0){
      y_true[i] = mu[times[i],id[i]] + if_else(cp1 < times[i], diff[cond[i]] , 0);
    }else{
      y_true[i] = mu[times[i],id[i]] ; 
    }
  }
}

model{
  
  // prior
  s_mu ~ cauchy(0, 2)T[0,];
  s_Y ~ cauchy(0, 2)T[0,];
  
  // model
  for(i in 2:N_time){
    for(j in 1:N_id){
      mu[i,j] ~ normal(mu[i-1, j], s_mu);
    }
  }
  
  for(i in 1:N){
    y[i] ~ normal(y_true[i], s_Y);
  }
}

generated quantities{
  real controlVSmirror[N_time];
  real controlVSstranger[N_time];
  real mirrorVSstranger[N_time];
  real control[N_time];
  real mirror[N_time];
  real stranger[N_time];
  
  for(i in 1:N_time){
    controlVSmirror[i] = if_else(cp1 < i, diff[1], 0);
    controlVSstranger[i] = if_else(cp1 < i, diff[2], 0);
    mirrorVSstranger[i] = if_else(cp1 < i, diff[1], 0) - if_else(cp1 < i, diff[2], 0);
    
    control[i] = mean(mu[i,]);
    mirror[i] = mean(mu[i,]) + if_else(cp1 < i, diff[1], 0); 
    stranger[i] = mean(mu[i,]) + if_else(cp1 < i, diff[2], 0); 
    
  }
}

