library(R.matlab)
getwd()
setwd('/media/steven/samsung_t5/fosr/papers/LFRM_Code')

design = readMat('Design.mat')
wt = design[[1]]
lbw = design[[2]]
sim_cov = design[[3]]
ni = design[[5]]
gest = design[[9]]
preeci = design[[10]]
