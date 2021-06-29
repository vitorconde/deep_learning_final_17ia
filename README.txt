# Dependencias do projeto em env.yml
# para importar dependencias basta executar
conda env install -f env.yml
conda env update -f env.yml

# Para atualizar as dependencias caso seja adicionado algum pacote novo basta rodar no prompt do anaconda o seguinte comando:
conda env export > env.yml