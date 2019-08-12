# Exemplo de Redes Neurais usando a biblioteca neuralnet e a base iris
# Comentado por Henrique Poyatos <poyatos@fiap.com.br>
# Extraído do curso "Formação Cientista de Dados com R e Python" 
# de Fernando Amaral e Jones Granatyr em
# https://www.udemy.com/cientista-de-dados/

# Instalação e carga da biblioteca neuralnet
install.packages("neuralnet")
library(neuralnet)

# Base iris
iris
# Cópia da base
myiris = iris

# Binarização das espécies
myiris = cbind(myiris, myiris$Species=='setosa')

head(myiris)

myiris = cbind(myiris, myiris$Species=='versicolor')
myiris = cbind(myiris, myiris$Species=='virginica')

# Teste da binarização
summary(myiris)

# Renomendo as novas colunas
names(myiris)[6] = 'setosa'
names(myiris)[7] = 'versicolor'
names(myiris)[8] = 'virginica'

summary(myiris)

# Gerando 150 números 1 e 2 aleatórios divididos em 70% e 30%
amostra = sample(2, 150, replace=T, prob=c(0.7,0.3))
amostra

# Separação entre base treino (70%) e base teste (30%)
myiristreino = myiris[amostra==1,]
myiristeste = myiris[amostra==2,]

# Criação do modelo de rede neural Multilayer Perceptron
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

# "Desbinarização" do resultado de teste
resultado$class = colnames(resultado[,1:3])[max.col(resultado[,1:3], ties.method='first')]

resultado$class

# Criação da matriz de confusão
confusao = table(resultado$class,myiristeste$Species)

confusao
# Percentual de acerto
sum(diag(confusao) * 100 / sum(confusao))