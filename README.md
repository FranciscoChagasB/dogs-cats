# Classificação de Gatos e Cachorros usando CNN

Este projeto demonstra como criar uma Convolutional Neural Network (CNN) para classificar imagens de gatos e cachorros. O código foi implementado em Python usando a biblioteca TensorFlow e é adequado para uma classificação binária de imagens.

## Como funciona o código

O código é projetado para treinar uma CNN com um conjunto de dados contendo imagens de treinamento e, em seguida, fazer previsões com base em novas imagens. Aqui está um resumo do que o código faz:

1. Importa as bibliotecas necessárias, incluindo TensorFlow.

2. Define o modelo da CNN, que consiste em camadas de convolução, camadas de max-pooling e camadas densas.

3. Compila o modelo, especificando o otimizador, a função de perda e as métricas.

4. Carrega as imagens de treinamento e teste usando um `ImageDataGenerator`, redimensionando-as para um tamanho específico e normalizando-as.

5. Treina o modelo com os dados de treinamento.

6. Avalia o desempenho do modelo com os dados de teste.

7. Faz previsões em novas imagens usando o modelo treinado.

8. Exibe o resultado da previsão (gato ou cachorro) com base na probabilidade retornada pela rede neural.

## DATASET

O dataset foi retirado do kaggle e está disponivel no link:
[DATASET](https://www.kaggle.com/datasets/stefancomanita/cats-and-dogs-40)

Certifique-se de ajustar os caminhos do código para refletir a estrutura das suas pastas.

## Requisitos

- Python
- TensorFlow
- Pillow (PIL fork) para processamento de imagens
