# Copilot Instructions - Análise de Consumo de Álcool

## Visão Geral do Projeto

Projeto educacional de ciência de dados sobre consumo de álcool **per capita** por país (WHO/FiveThirtyEight 2010). Análise completa em **português brasileiro** cobrindo EDA, visualizações geográficas interativas, testes de hipótese e modelagem preditiva (regressão e classificação).

## Arquitetura e Estrutura

- **Notebook principal**: `alcohol_consumption.ipynb` - fonte única com 10 seções numeradas (não há scripts .py)
- **Dados**: `drinks.csv` - 195 países × 5 colunas (country, beer_servings, spirit_servings, wine_servings, total_litres_of_pure_alcohol)
- **Documentação**: `README (2).md` - metadados do dataset original

### Fluxo de Análise (Seções do Notebook)
1. Importação de Bibliotecas → 2. Carregamento → 3. EDA → 4. Transformação → 5. Visualização Geográfica → 6. Testes de Hipótese → 7-8. Modelagem (Regressão/Classificação) → 9. Tuning → 10. Conclusões

**CRÍTICO**: Nunca renumere as seções principais. Use subseções (ex: `### 5.3`) para novos conteúdos.

## Convenções Críticas

### Nomenclatura Imutável
- DataFrame global: `df_drinks` (nunca renomear)
- Coluna de categoria: `consumption_category` (tipo `pd.Categorical`)
- Labels: `['Very Low', 'Low', 'Medium', 'High', 'Very High']` (ordem fixa, inglês)
- Variável auxiliar: `category_order` (lista das labels na ordem correta para Plotly)

### Sistema de Categorização (Bins Fixos)
```python
bins = [0, 1, 4, 7, 10, float('inf')]
labels = ['Very Low', 'Low', 'Medium', 'High', 'Very High']
df_drinks['consumption_category'] = pd.cut(
    df_drinks['total_litres_of_pure_alcohol'], 
    bins=bins, labels=labels, include_lowest=True
)
```
**NÃO altere os bins** - baseados em quartis dos dados de 2010. Alterar quebra comparabilidade.

## Stack de Tecnologia

### Bibliotecas Obrigatórias (Seção 1)
```python
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
import plotly.express as px
from scipy.stats import ttest_ind
from sklearn.model_selection import train_test_split, cross_val_score, RandomizedSearchCV
from sklearn.preprocessing import PolynomialFeatures, StandardScaler
from sklearn.linear_model import LinearRegression, Ridge, LogisticRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score, ...
from sklearn.naive_bayes import GaussianNB
from sklearn.pipeline import Pipeline
from statsmodels.stats.outliers_influence import variance_inflation_factor
```

### Divisão de Responsabilidades
- **Visualização geográfica**: Plotly Express exclusivo (mapas coropléticos)
- **Gráficos estatísticos**: Matplotlib + Seaborn (scatter plots, heatmaps, histogramas)
- **Testes de hipótese**: SciPy (`ttest_ind` para comparar grupos)
- **Modelagem**: scikit-learn (regressão, classificação, pipelines, tuning)

## Padrões de Implementação

### Mapas Coropléticos com Plotly (Seção 5)

**Mapa Categórico (discrete)** - Padrão para `consumption_category`:
```python
category_order = ['Very Low', 'Low', 'Medium', 'High', 'Very High']
fig = px.choropleth(
    df_drinks,
    locations="country",
    locationmode='country names',
    color="consumption_category",
    color_discrete_sequence=px.colors.sequential.Reds,
    category_orders={'consumption_category': category_order},  # Ordem OBRIGATÓRIA
    hover_data={
        'total_litres_of_pure_alcohol': ':.2f',  # 2 casas decimais
        'beer_servings': True,
        'wine_servings': True,
        'spirit_servings': True,
        'consumption_category': True,
        'category_numeric': False  # Ocultar se existir
    },
    title='... - Per Capita',
    labels={'consumption_category': 'Categoria de Consumo', ...}
)
fig.update_layout(geo=dict(projection_type='natural earth'), height=600, width=1000)
fig.show()
```

**Mapa Contínuo (continuous)** - Para valores numéricos diretos:
```python
fig2 = px.choropleth(
    df_drinks,
    color="total_litres_of_pure_alcohol",
    color_continuous_scale='Reds',  # Não discrete!
    # ... mesmas convenções de hover/labels
)
```

**Convenções de hover**: Sempre incluir total_litres (:.2f) + beer/wine/spirit servings. Sempre usar rótulos em português nas labels.

### Testes de Hipótese (Seção 6)

**Padrão para teste t independente** (ex: países islâmicos vs. não-islâmicos):
```python
# 1. Criar lista de países do grupo
islamic_countries = ["Afghanistan", "Pakistan", ...]
df_drinks["is_islamic"] = df_drinks["country"].isin(islamic_countries)

# 2. Separar grupos
group1 = df_drinks[df_drinks["is_islamic"]]['total_litres_of_pure_alcohol']
group2 = df_drinks[~df_drinks["is_islamic"]]['total_litres_of_pure_alcohol']

# 3. Executar teste t
result = ttest_ind(group1, group2)
t_stat, p_value = result.statistic, result.pvalue

# 4. Calcular estatísticas descritivas
mean1, std1, n1 = group1.mean(), group1.std(), len(group1)
mean2, std2, n2 = group2.mean(), group2.std(), len(group2)

# 5. Calcular graus de liberdade (manual se necessário)
# 6. Imprimir resultados formatados
```
**Sempre** incluir: médias dos grupos, estatística t, p-valor, conclusão em linguagem clara.

### Modelagem Preditiva (Seções 7-8)

**Função padronizada para avaliação de regressão**:
```python
def avaliar_regressao(modelo, X_tr, X_te, y_tr, y_te, nome):
    cv_r2 = cross_val_score(modelo, X_tr, y_tr, cv=5, scoring='r2').mean()
    modelo.fit(X_tr, y_tr)
    pred = modelo.predict(X_te)
    mae = mean_absolute_error(y_te, pred)
    rmse = mean_squared_error(y_te, pred, squared=False)
    r2 = r2_score(y_te, pred)
    return {
        'modelo': nome,
        'MAE': round(mae, 2),
        'RMSE': round(rmse, 2),
        'R2_teste': round(r2, 3),
        'R2_CV (média k=5)': round(cv_r2, 3)
    }
```

**Função padronizada para classificação**:
```python
def avaliar_classificacao(modelo, X_tr, X_te, y_tr, y_te, nome):
    modelo.fit(X_tr, y_tr)
    preds = modelo.predict(X_te)
    if hasattr(modelo, "predict_proba"):
        probas = modelo.predict_proba(X_te)[:, 1]
    else:
        probas = None
    acc = (preds == y_te).mean()
    prec = classification_report(y_te, preds, output_dict=True)['weighted avg']['precision']
    rec = classification_report(y_te, preds, output_dict=True)['weighted avg']['recall']
    f1 = classification_report(y_te, preds, output_dict=True)['weighted avg']['f1-score']
    auc = roc_auc_score(y_te, probas) if probas is not None else np.nan
    return {
        'modelo': nome,
        'acuracia': round(acc, 3),
        'precisao': round(prec, 3),
        'recall': round(rec, 3),
        'f1': round(f1, 3),
        'auc': round(auc, 3)
    }, preds, probas
```
**Retorna**: dict de métricas + predições + probabilidades (para ROC/Confusion Matrix posteriores).

**Preparação de dados para classificação binária** (Seção 8):
```python
df_class = df_drinks.dropna(subset=['consumption_category']).copy()
df_class['high_consumption'] = df_class['consumption_category'].isin(['High', 'Very High']).astype(int)

X_cls = df_class[['beer_servings', 'wine_servings', 'spirit_servings']]
y_cls = df_class['high_consumption']

# Baseline: predizer sempre a classe majoritária
baseline_acc = max(yc_train.mean(), 1 - yc_train.mean())
```
**Target**: `high_consumption` (binário 0/1) derivado de `consumption_category`.

**Split padrão**: `test_size=0.2, random_state=42`. Para classificação, adicionar `stratify=y_cls`.

**Pipelines para modelos complexos**:
```python
poly2 = Pipeline([
    ('poly', PolynomialFeatures(degree=2, include_bias=False)),
    ('scaler', StandardScaler()),
    ('model', LinearRegression())
])
```

### Tuning de Hiperparâmetros (Seção 9)

**Padrão para RandomizedSearchCV**:
```python
# 1. Definir grid de parâmetros
ridge_param = {
    'model__alpha': np.logspace(-3, 3, 20)
}

# 2. Criar pipeline com StandardScaler + modelo
ridge_pipe = Pipeline([
    ('scaler', StandardScaler()),
    ('model', Ridge())
])

# 3. Configurar busca
ridge_search = RandomizedSearchCV(
    ridge_pipe, 
    ridge_param, 
    n_iter=10, 
    cv=5, 
    scoring='r2', 
    random_state=42
)

# 4. Fit e extrair melhor modelo
ridge_search.fit(X_train, y_train)
best_alpha = ridge_search.best_params_['model__alpha']
best_ridge_rmse = avaliar_regressao(ridge_search.best_estimator_, ...)

# 5. Comparar com modelos base em resultados_reg
```

**Importante**: Sempre use `random_state=42` em RandomizedSearchCV para reprodutibilidade.

### Visualizações de Classificação (Seção 8)

**Matriz de Confusão padrão**:
```python
cm = confusion_matrix(yc_test, preds_log)
plt.figure(figsize=(4,3))
sns.heatmap(cm, annot=True, fmt='d', cmap='Reds', cbar=False)
plt.xlabel('Predito')
plt.ylabel('Real')
plt.title('Matriz de confusão - Logistic Regression')
plt.show()
```

**Curva ROC padrão**:
```python
fpr, tpr, _ = roc_curve(yc_test, prob_log)
plt.figure(figsize=(5,4))
plt.plot(fpr, tpr, color='firebrick', label=f'AUC = {roc_auc_score(yc_test, prob_log):.2f}')
plt.plot([0,1], [0,1], color='gray', linestyle='--')
plt.xlabel('Falso positivo')
plt.ylabel('Verdadeiro positivo')
plt.title('Curva ROC - Logistic Regression')
plt.legend()
plt.show()
```
**Sempre** use `color='firebrick'` ou `color='darkblue'` para consistência visual com paleta Reds.

### Transformações de Dados (Seção 4)

**Padrão de documentação em três células**:
1. Célula markdown explicando o PORQUÊ da transformação (critérios, lógica)
2. Célula código aplicando a transformação
3. Célula código verificando resultado (`.value_counts()`, `.head()`, etc.)

Exemplo:
```markdown
### 4.1 Criação de Categorias de Consumo
[Explicação dos critérios e bins]
```
```python
# Código de transformação
```
```python
# Verificação
df_drinks['consumption_category'].value_counts().sort_index()
```

### Análise de Multicolinearidade (Seção 7)

**Cálculo de VIF (Variance Inflation Factor)** antes de regressão múltipla:
```python
from statsmodels.stats.outliers_influence import variance_inflation_factor

vif_df = pd.DataFrame()
vif_df['variável'] = features.columns
vif_df['VIF'] = [variance_inflation_factor(features.values, i) 
                 for i in range(len(features.columns))]
```
**Interpretação**: VIF > 10 indica multicolinearidade problemática. No dataset atual, beer/wine/spirits têm VIF baixo (<3), indicando que regressão múltipla é válida.

### Análise de Scatter Plots (Seção 5.3)

**Função customizada para gráficos de dispersão** com labels de países:
```python
def simple_scatter(x, label_x):
    plt.figure(figsize=(10,6))
    plt.scatter(df_drinks[x], df_drinks["total_litres_of_pure_alcohol"], 
                color="darkblue", alpha=0.6)
    for i, row in df_drinks.iterrows():
        plt.text(row[x], row["total_litres_of_pure_alcohol"], 
                 row["country"], fontsize=7)
    plt.title(f"{label_x} vs Total Alcohol Consumption")
    plt.xlabel(label_x)
    plt.ylabel("Total litres of pure alcohol (per person/year)")
    plt.grid(alpha=0.3)
    plt.show()

simple_scatter("beer_servings", "Beer Servings")
```

## Dados e Limitações

### Estrutura do Dataset
- **195 linhas** (países), **5 colunas** (1 categórica: country; 4 numéricas)
- **Sem valores nulos**, sem duplicatas
- **Unidades**: servings = doses/pessoa/ano; total_litres = litros álcool puro/pessoa/ano
- **Ano**: 2010 (dados estáticos, sem série temporal)

### Outliers Conhecidos
- **Maior consumo**: Belarus (14.4 L/ano), Lithuania, Andorra (países da Europa Central/Oriental)
- **Consumo zero**: Afghanistan, Bangladesh, North Korea (fatores religiosos/políticos)
- **Outliers nos servings**: Cook Islands (spirit_servings=254), Czech Republic (beer_servings=361)

### Compatibilidade de Nomes de Países
A coluna `country` usa nomes compatíveis com `locationmode='country names'` do Plotly:
- "Antigua & Barbuda" (com &, não "and")
- "Cote d'Ivoire" (sem acento no dataset)
- "DR Congo" (não "Democratic Republic of the Congo")

**Se adicionar dados**: Verificar se nomes batem com a base geográfica do Plotly, senão países ficam vazios no mapa.

## Workflow de Desenvolvimento

### Ao Adicionar Novas Análises
1. Identifique a seção mais apropriada (5-10)
2. Crie subseção numerada (ex: `### 7.3`)
3. Preceda código com markdown explicativo
4. Use células separadas para código e verificação
5. Sempre termine com célula markdown interpretando resultados

### Debugging de Mapas
Se países aparecem vazios no mapa:
- Verifique `locationmode='country names'` (não use ISO codes)
- Compare nomes em `df_drinks['country']` com lista oficial do Plotly
- Use `fig.show()` para renderizar (não apenas `fig`)

### Executando o Notebook
```bash
# Instalar dependências (célula com pip install)
pip install pandas seaborn matplotlib plotly numpy scipy scikit-learn statsmodels jupyterlab

# Executar sequencialmente da seção 1 à 10
# Kernel deve ser reiniciado se houver mudanças no DataFrame global
```

## Contexto de Análise

### Hipóteses Investigadas
- **Hipótese I**: Países islâmicos consomem menos álcool (confirmada, p-valor ≪ 0.05)
- **Hipótese II**: Beer explica melhor o consumo total que wine/spirits (confirmada por correlações e R² de regressão simples)

### Padrões Descobertos
- **Geográficos**: Europa Central/Oriental e Rússia = consumo alto; Oriente Médio/Norte da África = muito baixo
- **Culturais**: Religião islâmica tem forte impacto negativo; clima frio correlaciona com destilados
- **Estatísticos**: Beer servings tem maior correlação com total_litres; wine é o preditor mais fraco

### Modelos Implementados
- **Regressão**: LinearRegression (simples/múltipla), PolynomialFeatures grau 2, Ridge (tuning)
- **Classificação**: GaussianNB, LogisticRegression (com/sem tuning de C/solver)
- **Métricas**: MAE, RMSE, R², CV-R², Acurácia, Precisão, Recall, F1, AUC-ROC

## Workflows Comuns

**Adicionar nova visualização geográfica**:
1. Criar subseção em `## 5` (ex: `### 5.3`)
2. Documentar o propósito/insight em markdown
3. Seguir padrão de `px.choropleth` com `locationmode='country names'`
4. Usar `color_discrete_sequence` ou `color_continuous_scale` com paleta `Reds` (consistência visual)

**Adicionar análise estatística**:
1. Inserir em nova subseção de `## 5` (EDA Gráfica) ou `## 3` (Exploração)
2. Preferir `seaborn` para gráficos estatísticos (histogramas, boxplots, correlações)
3. Usar configuração padrão: `sns.set_style()` se necessário customizar tema

**Adicionar novo teste de hipótese**:
1. Criar subseção em `## 6` (ex: `### 6.3`)
2. Markdown explicando H0, H1 e motivação
3. Código de preparação (criar grupos, flags, filtros)
4. Executar teste estatístico (`ttest_ind`, correlação, etc.)
5. Markdown com interpretação detalhada e conclusão prática

**Adicionar novo modelo preditivo**:
1. Definir problema (regressão seção 7, classificação seção 8)
2. Preparar dados com split padrão (`test_size=0.2, random_state=42`)
3. Usar funções `avaliar_regressao()` ou `avaliar_classificacao()`
4. Armazenar resultados em `resultados_reg` ou `resultados_cls`
5. Comparar modelos com DataFrame de resultados

## Estado Atual do Projeto

- **Última seção completa**: 8 (Classificação com GaussianNB + LogisticRegression)
- **Seções implementadas**: 1-Libs → 2-Load → 3-EDA → 4-Transform → 5-Viz Geo → 6-Hipóteses → 7-Regressão → 8-Classificação
- **Próximas análises sugeridas**: Seção 9 (Tuning de hiperparâmetros), Seção 10 (Conclusões finais)
- **Variáveis kernel principais**: 
  - DataFrames: `df_drinks`, `df_class`, `df_numeric`
  - Features/Target regressão: `features`, `target`, `X_train`, `X_test`, `y_train`, `y_test`
  - Features/Target classificação: `X_cls`, `y_cls`, `Xc_train`, `Xc_test`, `yc_train`, `yc_test`
  - Modelos: `nb_model`, `log_model`, `reg_beer`, `reg_multi`, `poly2`
  - Resultados: `resultados_reg` (DataFrame), `resultados_cls` (lista de dicts)

## Avisos Críticos de Edição

- **NUNCA** renumere seções principais (1-10) - apenas adicione subseções (ex: 7.3, 8.2)
- **NUNCA** modifique `bins` ou `labels` - são baseados em análise quartil específica dos dados de 2010
- **NUNCA** altere `random_state=42` nos splits - garante reprodutibilidade
- **SEMPRE** use `df_drinks` como nome do DataFrame principal - código existente depende disso
- **SEMPRE** inclua célula markdown explicativa ANTES de código de análise
- **SEMPRE** teste visualizações com `.show()` para Plotly ou `plt.show()` para matplotlib
- **SEMPRE** armazene resultados de modelos em `resultados_reg`/`resultados_cls` para comparação
- **SEMPRE** use `stratify=y_cls` ao dividir dados de classificação para balanceamento
