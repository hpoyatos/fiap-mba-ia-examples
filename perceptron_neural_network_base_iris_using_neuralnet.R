# Exemplo de Redes Neurais usando a biblioteca neuralnet e a base iris
# Comentado por Henrique Poyatos <poyatos@fiap.com.br>
# Extra�do do curso "Forma��o Cientista de Dados com R e Python" 
# de Fernando Amaral e Jones Granatyr em
# https://www.udemy.com/cientista-de-dados/

# Instala��o e carga da biblioteca neuralnet
install.packages("neuralnet")
library(neuralnet)

# Base iris
iris
# C�pia da base
myiris = iris

# Binariza��o das esp�cies
myiris = cbind(myiris, myiris$Species=='setosa')

head(myiris)

myiris = cbind(myiris, myiris$Species=='versicolor')
myiris = cbind(myiris, myiris$Species=='virginica')

# Teste da binariza��o
summary(myiris)

# Renomendo as novas colunas
names(myiris)[6] = 'setosa'
names(myiris)[7] = 'versicolor'
names(myiris)[8] = 'virginica'

summary(myiris)

# Gerando 150 n�meros 1 e 2 aleat�rios divididos em 70% e 30%
amostra = sample(2, 150, replace=T, prob=c(0.7,0.3))
amostra

# Separa��o entre base treino (70%) e base teste (30%)
myiristreino = myiris[amostra==1,]
myiristeste = myiris[amostra==2,]

# Cria��o do modelo de rede neural Multilayer Perceptron
modelo = neuralnet(setosa + versicolor + virginica ~ Sepal.Length + Sepal.Width + Petal.Length + Petal.Width, myiristreino, hidden=c(5,4))

# Plotagem da rede neural
modelo
plot(modelo)

# Uso da rede neural para reduzir os 30% de teste
teste = compute(modelo, myiristeste[,1:4])

# Resultado
teste$net.result

resultado = as.data.frame(teste$net.result)

names(resultado)[1] = 'setosa'
names(resultado)[2] = 'versicolor'
names(resultado)[3] = 'virginica'

head(resultado)

# "Desbinariza��o" do resultado de teste
resultado$class = colnames(resultado[,1:3])[max.col(resultado[,1:3], ties.method='first')]

resultado$class

# Cria��o da matriz de confus�o
confusao = table(resultado$class,myiristeste$Species)

confusao
# Percentual de acerto
sum(diag(confusao) * 100 / sum(confusao))