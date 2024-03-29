# Exemplo de Redes Neurais usando a H2o para reconhecimento de imagens
# Adaptado e comentado por Henrique Poyatos <poyatos@fiap.com.br>
# Extra�do do curso "Forma��o Cientista de Dados com R e Python" 
# de Fernando Amaral e Jones Granatyr em
# https://www.udemy.com/cientista-de-dados/

# Carregar os d�gitos da base treino para explora��o
digitos = read.csv(gzfile(file.choose()), header=F)

dim(digitos)
head(digitos)

# unlist() transforma obj em vector
# matrix() transforma o vector e matriz de novo 28x28 carregado por coluna
dig = t(matrix(unlist(digitos[20, -785]), nrow=28,byrow=F))
dig = t(apply(dig,2,rev))

# Plotar o d�gito
image(dig, col=grey.colors(255))
# Verifica��o do d�gito
digitos[20,785]

# Instalar e carregar o H2O (liberar no firewall, ~120MB)
# Precisa do Java instalado, testar nos labs!
install.packages("h2o")
library(h2o)
h2o.init()

# Carga da base treino
treino = h2o.importFile(file.choose())
dim(treino)

# Carga da base teste
teste = h2o.importFile(file.choose())
dim(teste)

# Transforma��o da coluna 785 (onde o n�mero realmente est� em Factor)
treino[,785] = as.factor(treino[,785])
teste[,785] = as.factor(teste[,785])

# Cria��o do modelo 
modelo = h2o.deeplearning(x=colnames(treino[,1:784]), y="C785", training_frame=treino, validation_frame=teste, distribution="AUTO", activation="RectifierWithDropout", hidden=c(64,64,64), sparse=TRUE, epochs=20)
# Plotagem do modelo
plot(modelo)

# Medi��o de performance do modelo?
h2o.performance(modelo)

# Previs�o de um n�mero
treino[20,785]
pred <- h2o.predict(model, newdata = treino[20,1:784])
pred$predict

#uso interno
options(warn=-1)