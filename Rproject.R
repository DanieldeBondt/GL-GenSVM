library(gensvm)
setwd('C:\\Users\\danie\\Documents\\BSc-Thesis')

data = read.table("breasttissueR.csv", header = FALSE)
#data = read.table("alcohol_consumptionR.csv", sep=";",header = FALSE)
x = data[, -10]
y = data[, 10]

print(nrow(x))
print(ncol(x))

fit = gensvm(x,y,epsilon= 1e-6, kappa=-0.95, p=2, lambda=2**-12, verbose=TRUE, random.seed=123)
V = coef(fit)
V

write.csv(V, "optimalV_R.csv")  
  