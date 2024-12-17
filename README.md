# Chronos Trader

Projeto da disciplina de deep learning do 2024.2 do Centro de Informática - UFPE

O objetivo da pesquisa foi avaliar se a utilização de tokens para treinar um agente de aprendizado por reforço poderia levar a uma melhor performance no investimento de ativos financeiros, portanto a proposta seria combinar uma LLM pré-treinada para séries temporais com um agente de DRL em uma ambiente de negociação de criptomoedas.

Composição do projeto:
* lab/docker-compose: cria uma aplicação admin para ver gráficos e métricas, mlflow para registrar e visualizar experimentos e tensorboard para registrar e visualizar treinamento dos modelos.
* lab/docker-compose-*: diferentes arquivos docker compose para criar o ambiente de backtesting para os diferentes modelos. O arquivo docker-compose-backtest-bh foi usado para os benchmarks que não envolveram o treinamento DRL.
* lab/docker: imagem para criar ambiente com freqtrade, mlflow e chronos.
* lab/user_data/admin: app admin
* lab/user_data/config: arquivos de configuração dos modelos e estratégias
* lab/user_data/data: dados das criptomoedas
* lab/user_data/freqaimodels: códigos dos modelos
* lab/user_data/strategies: códigos das estratégias


Para realizar um backtest:
1. cd lab
2. docker compose -f <nome_do_arquivo>.yaml up
