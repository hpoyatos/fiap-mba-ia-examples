# Exemplo de Redes Neurais usando a H2o para reconhecimento de imagens
# Adaptado e comentado por Henrique Poyatos <poyatos@fiap.com.br>
# Extraído do curso "Formação Cientista de Dados com R e Python" 
# de Fernando Amaral e Jones Granatyr em
# https://www.udemy.com/cientista-de-dados/

# Carregar os dígitos da base treino para exploração
digitos = read.csv(gzfile(file.choose()), header=F)

dim(digitos)
head(digitos)

# unlist() transforma obj em vector
# matrix() transforma o vector e matriz de novo 28x28 carregado por coluna
dig = t(matrix(unlist(digitos[20, -785]), nrow=28,byrow=F))
dig = t(apply(dig,2,rev))

# Plotar o dígito
image(dig, col=grey.colors(255))
# Verificação do dígito
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

# Transformação da coluna 785 (onde o número realmente está em Factor)
treino[,785] = as.factor(treino[,785])
teste[,785] = as.factor(teste[,785])

# Criação do modelo 
modelo = h2o.deeplearning(x=colnames(treino[,1:784]), y="C785", training_frame=treino, validation_frame=teste, distribution="AUTO", activation="RectifierWithDropout", hidden=c(64,64,64), sparse=TRUE, epochs=20)
# Plotagem do modelo
plot(modelo)

# Medição de performance do modelo?
h2o.performance(modelo)

# Previsão de um número
treino[20,785]
pred <- h2o.predict(model, newdata = treino[20,1:784])
pred$predict

#uso interno
options(warn=-1)