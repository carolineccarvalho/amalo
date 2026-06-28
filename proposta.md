# Geometria Local da Função de Perda e Generalização de Modelos de Aprendizado de Máquina

**Autores:** Caroline Campos Carvalho, Deborah Brito Yamamoto, Iago Zagnoli Albergaria, Manuel Junio Ferraz Cardoso, Marcos Daniel Souza Netto

**Data:** Abril de 2026

---

## 1. Introdução

Na área de aprendizado de máquina, um dos desafios centrais durante o treinamento de modelos é o *overfitting* (sobreajuste). Esse fenômeno ocorre quando o modelo se ajusta excessivamente aos dados de treinamento, chegando a "memorizá-los", o que compromete severamente sua capacidade de generalização e resulta em predições imprecisas para conjuntos de dados não vistos. O objetivo fundamental do aprendizado é garantir que o modelo extraia os padrões subjacentes reais do problema.

Neste contexto, a geometria local da função de perda desempenha um papel crucial. O projeto propõe a utilização da métrica de *sharpness* do mínimo local para quantificar a suscetibilidade de um modelo ao *overfitting*. O *sharpness* atua como uma medida de sensibilidade geométrica, indicando como a função de perda se comporta sob pequenas perturbações nos parâmetros do modelo. Estudos recentes demonstram a forte correlação entre a densidade dos autovalores da Hessiana e a otimização de redes neurais [Ghorbani et al., 2019], bem como a influência dessa métrica na estabilidade do treinamento com grandes *batches* [Yao et al., 2020].

A matriz Hessiana ($H$) descreve a curvatura local da função de perda. Uma estratégia amplamente adotada para quantificar o *sharpness* é calcular o seu maior autovalor, $\lambda_{max}(H)$. Um valor elevado de $\lambda_{max}(H)$ indica um mínimo agudo (*sharp minimum*), no qual pequenas alterações nos pesos causam grandes variações na perda, sugerindo a memorização dos dados. Em contrapartida, um $\lambda_{max}(H)$ reduzido caracteriza um mínimo plano (*flat minimum*). Modelos que convergem para mínimos planos tendem a ser mais robustos a ruídos e generalizam melhor para novos dados, conceito que fundamenta métodos modernos de otimização voltados para a generalização, como o *Sharpness-Aware Minimization* (SAM) [Foret et al., 2020].

O desafio prático reside na estimação de $\lambda_{max}(H)$. Em redes neurais modernas de alta dimensão, instanciar explicitamente a matriz Hessiana possui um custo computacional proibitivo. Para contornar este gargalo, serão utilizadas técnicas eficientes de aproximação, avaliando posteriormente as métricas obtidas em conjuntos de dados com diferentes níveis de complexidade, a fim de validar a correlação teórica entre a aproximação do autovalor máximo e o *overfitting*.

## 2. Metodologia

Esta seção descreve a metodologia técnica inicial adotada no projeto. O objetivo central é o uso de arquiteturas de redes neurais, como um *Multilayer Perceptron* (MLP), que serão treinadas para tarefas específicas utilizando o conjunto de dados detalhado em subseções posteriores. Ao final do processo de treinamento, a função de perda e os parâmetros convergidos serão utilizados para avaliar o *sharpness* do mínimo sobre a matriz Hessiana implícita $H$ sobre os parâmetros encontrados. A análise focará na extração do maior autovalor de $H$ para indicar o nível de *overfitting* e compará-lo à métrica do *gap de generalização*.

Para estabelecer uma base de comparação empírica, os modelos serão intencionalmente configurados e treinados de formas distintas, visando induzir ou evitar o sobreajuste. Também há a possibilidade de utilizar diferentes métodos de otimização que se comportam de maneira distinta, como o Adam (*Adaptive Moment Estimation*) e o SAM.

Para o cálculo do autovalor dominante $\lambda_{max}(H)$, será empregado o Método da Potência. Como a matriz Hessiana é excessivamente grande em modelos de alta dimensão, o método proposto por Pearlmutter [Pearlmutter, 1994] será utilizado, permitindo calcular o produto matriz-vetor da Hessiana de forma rápida e exata, sem a necessidade de instanciar a matriz completa. Essa técnica baseia-se na diferenciação automática e nas propriedades da derivada direcional [Pearlmutter, 1994]. O processo é dividido em dois passos principais:

1. O gradiente é computado convencionalmente por diferenciação automática;
2. Calcula-se a derivada direcional desse gradiente em relação a um vetor específico, reduzindo drasticamente o custo computacional e o consumo de memória.

## 3. Datasets

A priori, os dois primeiros *datasets* podem ser utilizados como uma espécie de gabarito a fim de validar a implementação dos métodos e modelos, uma vez que são conjuntos de dados pequenos que gerarão modelos com poucos parâmetros. Além disso, a proposta envolve implementar dois modelos para validação, testados sobre três possíveis *datasets* de diferentes domínios:

* **Titanic [dataset_titanic]:** Um problema clássico de dados tabulares focado em prever a sobrevivência de passageiros do navio com base em características demográficas e socioeconômicas (como sexo, faixa etária e classe da passagem).
* **Iris [dataset_iris]:** Um conjunto de dados voltado para a classificação multiclasse. O objetivo é desenvolver um modelo capaz de prever qual é a espécie da flor (dentre três possíveis) com base em suas características morfológicas (comprimento e largura de pétalas e sépalas).
* **MagnaTagATune [dataset_music]:** Um banco de dados que contém faixas de áudio musical e seus respectivos gêneros. A aplicação deste *dataset* consistiria em um modelo de categorização que, dado um ritmo ou características acústicas, classifica o estilo musical correspondente.

---

### Mapeamento de Citações (Chaves do BibTeX)

* `ghorbani2019investigation`: Ghorbani et al., 2019
* `yao2020hessian`: Yao et al., 2020
* `foret2020sharpness`: Foret et al., 2020
* `pearlmutter1994fast`: Pearlmutter, 1994
* `dataset_titanic`: Referência do Dataset Titanic
* `dataset_iris`: Referência do Dataset Iris
* `dataset_music`: Referência do Dataset MagnaTagATune
