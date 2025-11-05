## introdução / contexto

curiosidade: fazer a máquina aprender com o erro

antes: classificador fixo, aprendia uma vez só

agora: sistema vivo, aprende com feedback

reforço = tentativa, erro e recompensa

## conceito central

aprende por tentativa e erro

recompensa: +1 acerto / -1 erro

cada e-mail é um estado

ações: produtivo / improdutivo

não recebe resposta pronta

descobre sozinho o que dá mais resultado

o Q-Learning treina em episódios
 cada episódio é uma varredura completa no conjunto de exemplos (ou uma sequência de interações)

## dentro de cada episódio o agente:

lê um exemplo (e-mail)

decide uma ação (produtivo/improdutivo)

recebe uma recompensa:

+1 se acertou

−1 se errou

o modelo aprende de duas formas

feedback humano direto e base de dados acumulada

feedback humano

cada clique em certo ou errado é uma lição nova

feedback vai pro csv_foruseres

entra no próximo treino automaticamente

agente ajusta a Q-Table com base na resposta do usuário

aprendizado vivo em tempo real

reforço imediato curto prazo

quanto mais interação mais esperto ele fica

## base de dados

o train.csv é o conhecimento consolidado

garante que o agente não esquece o que já aprendeu

toda rodada de treino revisita a base completa

reforço acumulado longo prazo

mantém equilíbrio entre aprender e relembrar

combinação de experiência antiga com novas correções

## aprendizado híbrido

feedback ensina o instante

base consolida o aprendizado

juntos o modelo evolui de forma contínua

o sistema aprende com o uso e se fortalece com os dados

um ciclo constante entre ação erro correção e memória

## transformação do modelo

começou com regressão logística (supervisionado)

desafio: sair do aprendizado fixo

ideia: sem professor, aprende com interação

entrou o Q-Learning

Q-Table salva no SQLite

## Diferença entre SQLite e CSV no projeto AutoU

CSV = base de dados brutos (exemplos de treino, feedback, rótulos).
É onde ficam os textos e as classificações que o agente usa como experiência de aprendizado.

SQLite = base de conhecimento aprendido (a Q-Table).
É onde o modelo grava os resultados do aprendizado — os valores Q(s,a) que representam o quanto vale cada decisão em cada estado. ou seja guarda a exploracao do qlearning oq ele aprendeu.

## implementação técnica

guarda Q(s,a) = valor da ação por estado

atualiza com fórmula do Q-Learning

estado = tamanho + keywords

política ε-greedy: explora ou aproveita

epsilon controla exploração

epsilon decai com o tempo

## dificuldades

Q não atualizava, sistema parecia parado

solução: atualizar e persistir no banco

integrar feedback humano em tempo real

salvar correções no csv_foruseres

feedback entra no próximo treino

## frontend ↔ backend

frontend simples (HTML + JS)

usuário marca “certo” ou “errado”

feedback vai pra API

modelo aprende com o uso real

agente vai ficando mais humano

## métricas e visualização

gera accuracy, precision, recall, f1

gráficos de recompensa e epsilon

epsilon ↓ = menos exploração, mais confiança

quero desacelerar o decay do epsilon

## recompensa por episódio  
soma das recompensas recebidas em um treino completo  
mostra se o agente está melhorando com o tempo  
se a curva sobe ele está aprendendo  
se estabiliza atingiu o ótimo  
se cai está desaprendendo ou explorando demais  

## didática / metodologia

construção por etapas

validar cada camada separada

mudar um fator e observar

aprendizado visual com gráficos

ensinar aprendendo

## oq eu aprendi 

RL é mais sobre feedback que fórmula

persistência e logging = essenciais

feedback humano é ruído + ouro

ML bom continua aprendendo depois da entrega

## conclusão

sistema virou um aluno digital

cada correção = nova aula

aprende com experiência

IA é evoluir com o erro

## frases de impacto

a recompensa é o professor

explorar ou aproveitar? dilema da inteligência

o erro é dado de treino

o humano ensina pelo exemplo

Q-Learning = transformar erro em conhecimento

cada clique deixa o agente mais esperto

aprendizado é sobre feedback, não só algoritmo
