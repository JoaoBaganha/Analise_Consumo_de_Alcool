Usar `Pipeline` com duas etapas:
1. `scaler`
2. `mlp`

### 17. Grade de hiperparâmetros

Montar uma grade com **pelo menos 12 combinações**, variando **no mínimo dois hiperparâmetros**.

Usar uma grade enxuta e estável, adequada ao tamanho do dataset.

### Grade mínima recomendada

```python
param_grid = {
    'mlp__hidden_layer_sizes': [(8,), (12,), (8, 4)],
    'mlp__activation': ['relu', 'tanh'],
    'mlp__alpha': [0.0001, 0.001],
    'mlp__solver': ['adam']
}

Isso gera 12 combinações:

3 × 2 × 2 × 1 = 12.
18. Scoring do Grid Search

Usar:

scoring='accuracy'.

Se houver forte desbalanceamento, permitir:

scoring='f1'.

Mas, por padrão, usar acurácia se as classes estiverem razoavelmente equilibradas.

19. Grid Search no cenário sem isIslamic

Executar GridSearchCV com:

pipeline;
param_grid;
validação cruzada estratificada;
n_jobs=-1.

Registrar:

melhor combinação;
melhor score médio de validação cruzada;
tabela com os resultados principais (cv_results_).
20. Avaliação final no teste — cenário sem isIslamic

Usar o melhor estimador encontrado para prever o conjunto de teste e registrar:

acurácia;
F1-score;
relatório de classificação;
matriz de confusão.
21. Grid Search no cenário com isIslamic

Repetir a mesma lógica do item 19.

22. Avaliação final no teste — cenário com isIslamic

Repetir a mesma lógica do item 20.

Comparações Obrigatórias
23. Comparação antes e depois do ajuste

Criar uma tabela com quatro linhas:

baseline sem isIslamic;
tunado sem isIslamic;
baseline com isIslamic;
tunado com isIslamic.

Colunas:

cenário;
tipo do modelo;
acurácia;
F1-score.
24. Comparação do efeito de isIslamic

Criar uma interpretação em Markdown respondendo:

a inclusão de isIslamic melhorou o baseline?
a inclusão de isIslamic melhorou o modelo tunado?
a variável parece agregar informação útil?
25. Identificação de overfitting / underfitting

Comparar:

score médio de validação cruzada;
score no treino, se necessário;
score no teste.

Adicionar uma análise curta sobre:

possível overfitting;
possível underfitting;
estabilidade do modelo.
Análise de Sensibilidade de Hiperparâmetros
26. Objetivo

Adicionar uma seção separada para a etapa individual.

A ideia é:

fixar a melhor configuração encontrada;
variar apenas um hiperparâmetro;
observar o impacto no desempenho.
27. Implementação da análise

Criar uma função genérica para:

receber um pipeline base;
receber o nome de um hiperparâmetro;
receber uma lista de 5 ou mais valores;
treinar e avaliar cada configuração;
registrar os resultados em um DataFrame.
28. Hiperparâmetro recomendado para análise

Implementar por padrão a análise de:

hidden_layer_sizes.

Testar pelo menos 5 valores, por exemplo:

[(4,), (8,), (12,), (16,), (8, 4)]

Opcionalmente deixar preparado para trocar por:

alpha;
max_iter;
activation.
29. Métrica da sensibilidade

Registrar, para cada valor:

score médio de validação cruzada;
acurácia no teste;
F1-score no teste.
30. Visualização

Criar pelo menos um gráfico comparativo usando matplotlib.

Sugestão:

eixo x = valores do hiperparâmetro;
eixo y = acurácia ou F1-score.
31. Interpretação

Adicionar célula Markdown respondendo:

qual valor teve melhor desempenho;
o hiperparâmetro causou grande ou pequena variação;
o modelo foi sensível a essa mudança.
Funções Auxiliares Esperadas

Criar funções para evitar repetição excessiva.

32. Função para avaliar modelo

Implementar uma função como:
evaluate_model(model, X_train, X_test, y_train, y_test, model_name, scenario_name)

Ela deve retornar:

acurácia;
F1-score;
classification report;
confusion matrix;
previsões.
33. Função para exibir resultados

Implementar função para:

imprimir métricas;
mostrar matriz de confusão;
retornar dicionário resumido para tabela final.
34. Função para sensibilidade

Implementar função como:
run_sensitivity_analysis(...)

Conteúdo em Markdown que deve existir no notebook
35. Explicação do dataset

Uma célula Markdown breve contendo:

origem dos dados;
número de instâncias;
atributos usados;
justificativa de uso no trabalho.
36. Explicação do pré-processamento

Explicar:

ausência/tratamento de nulos;
separação treino/teste;
padronização com StandardScaler.
37. Explicação do baseline

Explicar que o baseline é o modelo inicial, sem otimização sistemática.

38. Explicação do Grid Search

Explicar que o Grid Search testa sistematicamente combinações de hiperparâmetros com validação cruzada para encontrar a melhor configuração.

39. Explicação da comparação com isIslamic

Explicar que o experimento busca avaliar se a inclusão dessa variável cultural/religiosa melhora o poder preditivo do modelo.

40. Explicação da análise de sensibilidade

Explicar que, após encontrar a melhor configuração, um hiperparâmetro é variado isoladamente para medir seu impacto.

O que deve ser removido ou reduzido do notebook antigo
41. Remover ou deixar secundário

Reduzir drasticamente ou remover do fluxo principal:

regressão linear;
ridge;
regressão polinomial;
análise de resíduos;
VIF;
testes de hipótese detalhados;
blocos longos de EDA que não contribuam para a classificação com MLP.
42. O que pode permanecer

Pode permanecer de forma resumida:

uma contextualização breve do dataset;
uma visualização simples da distribuição do alvo;
a construção da variável isIslamic;
a criação da variável de classificação.
Saídas Finais Esperadas
43. Tabelas finais

O notebook deve terminar com:

tabela comparativa dos quatro cenários principais;
tabela dos melhores hiperparâmetros encontrados;
tabela da análise de sensibilidade.
44. Conclusão final

Adicionar uma seção final em Markdown respondendo explicitamente:

O modelo melhorou após o ajuste de hiperparâmetros?
Qual combinação teve melhor desempenho?
A variável isIslamic ajudou?
Houve sinais de overfitting ou underfitting?
O hiperparâmetro analisado teve impacto relevante?
Que conclusão geral pode ser tirada sobre o comportamento da MLP nesse dataset?
Requisitos de Qualidade do Código
45. Boas práticas
código limpo e comentado;
células com ordem lógica;
sem duplicação desnecessária;
nomes de variáveis claros;
reprodutível com random_state=42.
46. Robustez
verificar se colunas esperadas existem;
falhar com mensagens claras caso algum campo essencial esteja ausente;
evitar hardcodes desnecessários além do nome do arquivo e nomes das colunas.
47. Compatibilidade

O notebook deve rodar com Python 3.10+ e versões comuns do scikit-learn.

Entregável esperado do Copilot

O GitHub Copilot deve:

refatorar o notebook existente;
preservar o dataset e a lógica central já aceita;
reorganizar o fluxo para classificação com MLP;
implementar os dois cenários:
sem isIslamic;
com isIslamic;
implementar baseline e Grid Search;
implementar análise de sensibilidade;
gerar tabelas e gráficos finais;
manter Markdown explicativo ao longo do notebook.
Observação final importante

O notebook final não deve parecer um projeto genérico de machine learning.

Ele deve parecer claramente um trabalho acadêmico construído especificamente para esta atividade, obedecendo aos requisitos da professora:

dataset de classificação;
MLP;
Grid Search;
validação cruzada;
mínimo de 12 combinações;
análise individual de hiperparâmetros;
comparação de desempenho;
interpretação dos resultados.