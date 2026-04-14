# Especificação para refatoração/criação do notebook da Parte 2 — SOM

## Objetivo geral

Refatorar o notebook existente da análise de consumo de álcool para atender **especificamente** ao segundo trabalho prático da disciplina, agora com foco em **Mapas Auto-Organizáveis (SOM - Self-Organizing Maps)**.

O notebook final deve reaproveitar o mesmo dataset do trabalho anterior e transformar o fluxo anterior em um estudo de **aprendizado não supervisionado**, com treinamento, avaliação, visualização e interpretação de agrupamentos usando SOM.

Este notebook **não deve parecer genérico**. Ele deve parecer um trabalho acadêmico construído para esta atividade, com continuidade clara em relação ao notebook anterior de MLP e obedecendo ao enunciado da professora.

---

## Referência principal de implementação

A **principal referência estrutural e técnica** deve ser o notebook de exemplo `SOM.ipynb` anexado pelo usuário.

Isso significa que a implementação deve seguir, sempre que possível, o mesmo estilo do notebook-base, especialmente em:

- uso da biblioteca `MiniSom`;
- padronização com `StandardScaler`;
- cálculo das métricas `quantization_error` e `topographic_error`;
- visualizações com `matplotlib`;
- mapas de ativação com contagem por neurônio;
- component planes a partir de `som.get_weights()`;
- estilo direto, didático e visual.

---

## Restrições de bibliotecas

### Bibliotecas permitidas e preferenciais
Usar prioritariamente apenas bibliotecas alinhadas ao notebook de referência `SOM.ipynb` e ao notebook do trabalho anterior:

```python
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from minisom import MiniSom
from sklearn.preprocessing import StandardScaler
from collections import defaultdict
```

### Bibliotecas opcionais
Podem ser usadas apenas se realmente forem necessárias e sem descaracterizar o notebook:

- `math`
- `itertools`
- `textwrap`

### Bibliotecas que devem ser evitadas
Não ampliar o stack com bibliotecas desnecessárias ou mais avançadas do que o notebook-base. Evitar, salvo necessidade absoluta:

- `plotly`
- `statsmodels`
- `scipy` para análises extras irrelevantes
- bibliotecas de clustering adicionais (`yellowbrick`, `umap`, `hdbscan`, `somoclu`, etc.)
- pipelines complexos do `sklearn`
- `GridSearchCV` para o SOM
- dashboards ou frameworks web

### Instalação
Se necessário, incluir apenas:

```python
!pip install minisom
```

---

## Dataset e continuidade com o trabalho anterior

O notebook deve **reaproveitar o mesmo dataset do trabalho anterior**, isto é, a base de consumo de álcool por país.

A lógica central já aceita no trabalho anterior deve ser preservada sempre que fizer sentido:

- leitura do dataset original;
- breve contextualização do problema;
- reaproveitamento das variáveis já consolidadas;
- possibilidade de reaproveitar a variável `isIslamic` ou equivalente, se ela já existir no notebook anterior;
- possibilidade de reaproveitar a variável de classificação criada anteriormente, mas **apenas para análise posterior**, nunca como alvo de treinamento do SOM.

### Ponto fundamental
No trabalho com SOM, o treinamento deve ser **não supervisionado**.

Portanto:

- a variável-alvo do trabalho anterior **não pode ser usada como target de treino**;
- rótulos antigos podem aparecer apenas para:
  - colorir visualizações;
  - interpretar agrupamentos;
  - comparar coerência com o problema original de classificação.

---

## Estrutura obrigatória do notebook

O notebook final deve ter uma sequência lógica, limpa e acadêmica.

### 1. Identificação da equipe e declaração de uso de IA
Logo no início deve haver uma seção em Markdown com:

- nomes dos integrantes da equipe;
- ferramenta de IA utilizada, se houve uso;
- finalidade do uso da IA;
- extensão aproximada da contribuição da IA;
- medidas adotadas para validação do conteúdo.

Essa seção é obrigatória conforme o enunciado.

### 2. Introdução breve
Explicar em Markdown:

- que esta é a segunda etapa do trabalho;
- que o mesmo dataset do MLP foi reaproveitado;
- que agora o objetivo é explorar padrões e agrupamentos com SOM;
- que a análise é não supervisionada.

### 3. Carregamento e reconhecimento do dataset
Mostrar:

- leitura do arquivo;
- `head()`;
- `info()`;
- `shape`;
- verificação simples de nulos e duplicados.

Essa parte deve ser objetiva. Não transformar novamente o notebook em um EDA extenso.

### 4. Justificativa do pré-processamento
Explicar claramente:

- quais colunas entram no SOM;
- quais colunas ficam apenas para identificação/interpretação;
- se houve exclusão de atributos;
- se houve inclusão da variável `isIslamic`;
- por que a padronização com `StandardScaler` foi utilizada.

### 5. Preparação dos dados
Implementar:

- seleção das features numéricas usadas no SOM;
- eventual criação ou reaproveitamento de `isIslamic`;
- padronização com `StandardScaler`;
- separação entre:
  - matriz usada no treinamento do SOM;
  - metadados auxiliares como país e classe original.

### 6. Configuração do SOM
Criar uma seção de hiperparâmetros com valores explícitos, como no notebook-base.

Deve conter pelo menos:

- `som_x`
- `som_y`
- `sigma`
- `learning_rate`
- `num_epochs`

Esses hiperparâmetros devem ser comentados de forma didática.

### 7. Treinamento
Treinar o SOM com `MiniSom`, seguindo o estilo do notebook de referência:

- inicialização do SOM;
- `random_weights_init`;
- `train`.

### 8. Avaliação do treinamento
Calcular e exibir obrigatoriamente:

- `Quantization Error`
- `Topographic Error`

Essas métricas devem ser explicadas em Markdown em linguagem simples.

### 9. Visualizações obrigatórias
O notebook deve conter, com títulos claros e interpretação:

- **U-Matrix**
- **Hit Map / mapa de ativação dos neurônios**
- **Component Planes**
- **Scatterplots por pares de atributos relevantes**

Se houver muitas variáveis, selecionar apenas as mais relevantes para os scatterplots.

### 10. Interpretação dos agrupamentos
Deve haver uma seção em Markdown respondendo:

- que agrupamentos o SOM sugere;
- se há padrões por intensidade de consumo;
- se há indícios de agrupamentos culturais;
- se os grupos encontrados fazem sentido em relação ao problema original.

### 11. Comparação breve com o trabalho anterior (MLP)
Adicionar uma seção curta em Markdown explicando a diferença entre:

- MLP como técnica supervisionada de classificação;
- SOM como técnica não supervisionada de organização e agrupamento.

A comparação deve ser conceitual e resumida.

### 12. Conclusão final
Responder explicitamente:

- o SOM conseguiu representar bem os dados?
- os agrupamentos foram coerentes?
- a variável cultural/religiosa ajudou na interpretação, se usada?
- como o SOM complementa o trabalho anterior com MLP?

---

## Cenários a implementar

O notebook deve preferencialmente implementar **dois cenários**, desde que isso não complique excessivamente o fluxo:

### Cenário 1 — sem `isIslamic`
Usar apenas os atributos originais de consumo.

### Cenário 2 — com `isIslamic`
Adicionar a variável `isIslamic` às features do SOM.

### Comparação entre cenários
Se os dois cenários forem implementados, comparar de forma objetiva:

- Quantization Error;
- Topographic Error;
- diferenças visuais observadas nos mapas;
- impacto interpretativo da variável adicional.

Se houver necessidade de simplificação, o notebook pode treinar apenas um cenário principal e deixar o segundo cenário como bloco complementar. Mas a preferência é pelos dois.

---

## Hiperparâmetros e experimentação

O enunciado pede ajuste dos hiperparâmetros do SOM.

### O que fazer
Testar algumas combinações razoáveis de:

- tamanho da malha;
- `sigma`;
- `learning_rate`;
- número de épocas.

### O que evitar
Não transformar isso em um framework sofisticado de busca automática. Evitar GridSearch formal.

### Forma recomendada
Fazer uma exploração leve e didática, por exemplo:

- testar 3 a 6 combinações manuais;
- registrar os resultados em uma tabela simples;
- escolher a melhor configuração com base principalmente em:
  - menor `Quantization Error`;
  - menor `Topographic Error`;
  - melhor legibilidade visual do mapa.

### Exemplo de tabela esperada
A tabela de comparação de hiperparâmetros pode conter colunas como:

- `som_grid`
- `sigma`
- `learning_rate`
- `epochs`
- `quantization_error`
- `topographic_error`

---

## Visualizações: requisitos detalhados

### 1. U-Matrix
Deve mostrar as distâncias entre neurônios vizinhos para sugerir fronteiras entre grupos.

Implementar com `som.distance_map()` e `matplotlib`.

### 2. Hit Map
Seguir o estilo do notebook de referência:

- agrupar amostras por neurônio vencedor (`winner` / BMU);
- usar tamanho do marcador para indicar quantidade de amostras;
- opcionalmente usar cor por classe dominante, caso exista uma variável de rótulo auxiliar.

### 3. Component Planes
Seguir o padrão do notebook de referência:

- usar `som.get_weights()`;
- gerar um mapa por variável;
- organizar em subplots;
- usar barra de cor.

### 4. Scatterplots
Produzir alguns scatterplots dos atributos mais relevantes, preferencialmente:

- `beer_servings` vs `total_litres_of_pure_alcohol`
- `wine_servings` vs `total_litres_of_pure_alcohol`
- `spirit_servings` vs `total_litres_of_pure_alcohol`

Esses gráficos podem ajudar a conectar os grupos observados no SOM com relações mais intuitivas do dataset.

### 5. Opcional — animação
Só incluir animação da evolução dos neurônios se isso já estiver simples e estável. Não é prioridade.

---

## Funções auxiliares esperadas

Criar funções simples para evitar repetição, sem exagerar na abstração.

### 1. Função para treinar e avaliar SOM
Implementar uma função semelhante a:

```python
def train_and_evaluate_som(X_scaled, som_x, som_y, sigma, learning_rate, num_epochs, random_seed=42):
    ...
```

Ela deve:

- criar o SOM;
- inicializar pesos;
- treinar;
- calcular `quantization_error`;
- calcular `topographic_error`;
- retornar o SOM e as métricas.

### 2. Função para tabela de experimentos
Implementar função para registrar resultados de diferentes hiperparâmetros em `DataFrame`.

### 3. Função para hit map
Criar uma função para plotar o mapa de ativação.

### 4. Função para component planes
Criar uma função para gerar os mapas de componentes com base no array de pesos.

Essas funções devem ser simples, legíveis e coerentes com o estilo acadêmico do notebook.

---

## Conteúdo em Markdown que deve existir no notebook

O notebook precisa conter explicações curtas e claras ao longo do fluxo.

### Explicações obrigatórias

#### 1. Sobre o dataset
Explicar:

- origem dos dados;
- número de instâncias;
- atributos principais;
- motivo de reaproveitamento do dataset.

#### 2. Sobre o pré-processamento
Explicar:

- tratamento ou ausência de nulos;
- escolha das variáveis;
- padronização com `StandardScaler`.

#### 3. Sobre o SOM
Explicar que o SOM é uma rede neural não supervisionada que projeta dados de alta dimensão em uma malha bidimensional preservando, na medida do possível, relações de vizinhança.

#### 4. Sobre Quantization Error
Explicar que mede o quanto os neurônios representam bem as amostras.

#### 5. Sobre Topographic Error
Explicar que mede o quanto a topologia dos dados foi preservada.

#### 6. Sobre U-Matrix
Explicar que ajuda a visualizar regiões de separação entre possíveis grupos.

#### 7. Sobre Component Planes
Explicar que mostram como cada variável se distribui ao longo do mapa.

#### 8. Sobre a comparação com a MLP
Explicar que o SOM não substitui a MLP; ele complementa a compreensão da estrutura dos dados.

---

## O que deve ser removido ou reduzido do notebook anterior

Reduzir drasticamente ou remover do fluxo principal:

- regressão linear;
- ridge;
- regressão polinomial;
- testes de hipótese longos;
- VIF;
- resíduos;
- classificadores alternativos irrelevantes;
- Grid Search do trabalho anterior;
- métricas de classificação como foco central;
- matrizes de confusão como resultado principal.

### O que pode permanecer resumidamente
Pode permanecer, de forma curta e funcional:

- contextualização do dataset;
- distribuição da classe original, se isso ajudar na interpretação;
- construção da variável `isIslamic`;
- criação da variável categórica do trabalho anterior, apenas para apoio analítico.

---

## Tabelas finais esperadas

O notebook deve terminar com tabelas bem organizadas.

### 1. Tabela de experimentos de hiperparâmetros
Com as combinações testadas para o SOM.

### 2. Tabela de métricas do cenário final
Com pelo menos:

- cenário;
- som grid;
- sigma;
- learning rate;
- epochs;
- quantization error;
- topographic error.

### 3. Se houver dois cenários
Criar uma tabela comparativa entre:

- sem `isIslamic`;
- com `isIslamic`.

---

## Requisitos de qualidade do código

### Boas práticas
O notebook deve ter:

- código limpo;
- comentários objetivos;
- células em ordem lógica;
- títulos claros;
- variáveis com nomes legíveis;
- reprodutibilidade com `random_seed` ou `random_state=42` quando aplicável.

### Robustez
O notebook deve:

- verificar se o arquivo foi carregado corretamente;
- verificar se colunas esperadas existem;
- falhar com mensagens claras caso alguma coluna essencial esteja ausente;
- evitar hardcodes desnecessários além do nome do arquivo e nomes das colunas essenciais.

### Compatibilidade
O notebook final deve rodar com:

- Python 3.10+
- versões comuns de `pandas`, `numpy`, `matplotlib`, `scikit-learn` e `minisom`

---

## Entregável esperado da IA

A IA deve:

- reaproveitar o dataset e a lógica central do trabalho anterior;
- reorganizar o notebook para SOM;
- manter o foco em aprendizado não supervisionado;
- implementar treinamento do SOM com hiperparâmetros explícitos;
- calcular `Quantization Error` e `Topographic Error`;
- gerar U-Matrix, Hit Map e Component Planes;
- produzir interpretação em Markdown ao longo do notebook;
- comparar de forma resumida com a etapa anterior de MLP;
- manter o notebook acadêmico, claro e específico para a atividade.

---

## Observação final importante

O notebook final deve demonstrar claramente que:

1. houve continuidade metodológica em relação ao trabalho anterior com MLP;
2. o mesmo dataset foi reaproveitado conscientemente;
3. o SOM foi utilizado para descobrir padrões e agrupamentos, e não para classificar diretamente;
4. as visualizações foram usadas como ferramenta analítica principal;
5. as métricas e os agrupamentos foram interpretados de forma crítica e conectados ao problema original do dataset.

