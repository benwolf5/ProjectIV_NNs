fun <- function(x){
  return (-1*(x-1)*(x-1))
}

x <- runif(30,0, 2)
noise <- runif(30,-0.1,0.1)
y<- fun(x) + noise +1


par(mfrow =c(1,3))

plot(x,y, col="blue", main = "Good Balance", cex.lab=2, cex.axis=2, cex.main=1.5, cex.sub=2)
curve(-(x-1)^2+1, add = TRUE)


plot(x,y, col="blue", main = "High Bias (Underfitting)", cex.lab=2, cex.axis=2, cex.main=1.5, cex.sub=2)
abline(a=0.75, 0)

plot(x,y, col="blue", main = "High Variance (Overfitting)", cex.lab=2, cex.axis=2, cex.main=1.5, cex.sub=2)


