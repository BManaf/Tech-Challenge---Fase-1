# Tech-Challenge---Fase-1
Classificação de Tumores Mamários Utilizando Técnicas de Machine Learning Supervisionado e Deep Learning
1. Apresentação do desafio

O Tech Challenge – Fase 1 propõe o desenvolvimento da base de um sistema inteligente capaz de auxiliar médicos e equipes clínicas na análise inicial de exames médicos e dados clínicos, considerando o crescimento do volume de pacientes, exames e prontuários digitais em ambientes hospitalares.

Hospitais universitários enfrentam desafios como alto volume de exames, necessidade de triagem rápida, redução de erros humanos e otimização do tempo dos profissionais de saúde. Nesse contexto, o desafio consiste na aplicação de fundamentos de Inteligência Artificial, Machine Learning e Visão Computacional para criar uma solução que processe dados médicos automaticamente e destaque informações relevantes para o diagnóstico.

O sistema proposto não substitui o médico, atuando exclusivamente como ferramenta de apoio à decisão clínica.

2. Objetivo geral do projeto

O objetivo deste projeto é desenvolver uma solução inicial de Inteligência Artificial capaz de processar dados médicos estruturados relacionados ao diagnóstico de câncer de mama, aplicar algoritmos de Machine Learning para classificação binária, realizar exploração, pré-processamento, modelagem e avaliação dos dados, interpretar os resultados de forma transparente, discutir limitações e viabilidade prática e, como atividade opcional, implementar um módulo de Visão Computacional aplicado a imagens médicas de câncer de mama.

3. Descrição da solução proposta

A solução foi estruturada em dois módulos complementares.

3.1 Módulo principal – Machine Learning com dados estruturados

O módulo principal realiza a classificação binária de exames médicos com foco no diagnóstico de câncer de mama, abrangendo todo o fluxo de um projeto de Machine Learning, desde a análise exploratória dos dados até a avaliação e interpretação dos modelos preditivos.

3.2 Módulo extra – Visão Computacional aplicada ao câncer de mama

Como extensão opcional do projeto, foi desenvolvido um módulo de Visão Computacional baseado em Redes Neurais Convolucionais (CNNs) para a classificação de imagens histopatológicas de câncer de mama. Esse módulo possui caráter demonstrativo e explora técnicas de Deep Learning aplicadas diretamente a exames de imagem, conforme sugerido no enunciado do desafio.

4. Bases de dados utilizadas
4.1 Dataset principal – Dados estruturados

Nome: Breast Cancer Wisconsin (Diagnostic)
Fonte: UCI Machine Learning Repository (acessado via scikit-learn)

Trata-se de um dataset público amplamente utilizado em pesquisas acadêmicas, contendo atributos extraídos de exames citológicos de nódulos mamários.

Características principais:

Tipo: dados tabulares

Domínio: diagnóstico oncológico

Objetivo: classificar tumores como benignos ou malignos

Variável alvo:

0 – benigno

1 – maligno

O dataset é carregado diretamente pela biblioteca scikit-learn por meio da função load_breast_cancer, garantindo reprodutibilidade e simplicidade de execução.

4.2 Dataset extra – Dados de imagem

Nome: Breast Cancer Histopathological Images (BreaKHis)
Fonte: Kaggle

https://www.kaggle.com/datasets/awsaf49/cbis-ddsm-breast-cancer-image-dataset/data

Esse conjunto de dados é composto por imagens histopatológicas de tecido mamário, organizadas em classes benignas e malignas. O dataset foi utilizado exclusivamente no desafio extra para demonstrar a aplicação de Redes Neurais Convolucionais no contexto do diagnóstico de câncer de mama por imagens médicas.

5. Organização e arquitetura do código

O projeto foi estruturado seguindo boas práticas de Engenharia de Software, Ciência de Dados e Machine Learning, priorizando clareza, modularidade e reprodutibilidade.

Estrutura geral do projeto:

codigo
  principal
  estudodados_cancer_mama.py      Análise exploratória dos dados
  principal_cancer_mama.py        Execução do pipeline de Machine Learning com dados estruturados

codigo
  desafio
    desafio_cancer_mama.py   Implementação da CNN para o desafio extra

data
  Bases de dados utilizadas (não versionadas)

6. Metodologia de desenvolvimento
6.1 Exploração dos dados (EDA)

A análise exploratória dos dados teve como objetivo compreender as características do dataset, incluindo estatísticas descritivas, distribuição das classes, análise de correlação entre variáveis e identificação de padrões relevantes para o diagnóstico de câncer de mama.

6.2 Pré-processamento dos dados

Foi implementado um pipeline de pré-processamento utilizando ferramentas do scikit-learn, incluindo padronização de variáveis numéricas e separação adequada entre conjuntos de treino e teste, evitando vazamento de dados.

6.3 Modelagem preditiva

Foram treinados e comparados três algoritmos de classificação supervisionada: Regressão Logística, Random Forest e K-Nearest Neighbors (KNN). Essa abordagem permite avaliar tanto modelos lineares quanto não lineares no contexto do diagnóstico médico.

7. Avaliação dos modelos

O desempenho dos modelos foi avaliado utilizando as métricas accuracy, recall, F1-score e ROC-AUC. Em aplicações médicas, métricas como recall e F1-score são especialmente relevantes, pois falsos negativos podem acarretar riscos clínicos significativos.

8. Interpretabilidade e transparência

Considerando a criticidade de aplicações em saúde, o projeto incorpora técnicas de interpretabilidade, incluindo análise de importância de variáveis e métodos baseados em SHAP, permitindo compreender como cada atributo influencia as decisões do modelo.

9. Limitações e considerações éticas

Apesar dos resultados obtidos, o projeto apresenta limitações importantes, como o uso de datasets acadêmicos e controlados, a ausência de validação clínica real e a impossibilidade de uso direto em ambiente hospitalar. O sistema deve ser interpretado exclusivamente como ferramenta de apoio à decisão clínica.

10. Considerações finais

Este projeto demonstra a aplicação prática de técnicas de Inteligência Artificial, Machine Learning e Deep Learning no suporte ao diagnóstico de câncer de mama, atendendo aos requisitos do Tech Challenge – Fase 1 e respeitando princípios técnicos, éticos e acadêmicos.
