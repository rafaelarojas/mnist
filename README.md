# Aplicação prática de Redes Convolucionais
Utilizar redes neurais convolucionais para tarefas de visão computacional.

![alt text](gif.gif)

É possível anexar a foto de um número, e os modelos convolucional e linear irão identificar qual número está na imagem. Além disso, tem-se também o tempo de inferência que cada modelo levou para identificar a imagem de input.

Para ver a demonstração completa, [clique aqui](https://drive.google.com/file/d/1BNCYyMlQ8iXJFrxA4G6YCcKuIYOgiWiS/view?usp=sharing)

---
### Modelos

Foi feita uma comparação dos modelos, considerando que ambos foram treinados com 100 épocas.

##### Modelo linear
|    Tempo de treinamento   |    Acurácia    |
| ------------------------- | -------------- |
| 227.295852184295 segundos | 0.984300017    |

##### Modelo convolucional
|    Tempo de treinamento   |    Acurácia    |
| ------------------------- | -------------- |
| 444.202180147171 segundos | 0.994499981    |


---
### Rotas

#### Rota princial

`GET /`
- Descrição: Renderiza a página inicial da aplicação, onde o usuário pode fazer upload de uma imagem para predição.
- Parâmetros de Consulta: Nenhum.
- Resposta: HTML - Renderiza o arquivo index.html.

#### Rota da predição com CNN

`POST /predict_cnn`
- Descrição: Recebe uma imagem de um dígito manuscrito e retorna a predição feita pelo modelo CNN.
- Parâmetros do Corpo da Requisição: file - Arquivo de imagem a ser enviado no formato multipart/form-data.
- Resposta:
```json

    {
        "digit_cnn": int,
        "inference_time_cnn": float
    }

```
    `digit_cnn`: Dígito predito pelo modelo CNN.
    `inference_time_cnn`: Tempo de inferência do modelo CNN em segundos.

- Exemplo de Requisição:
```
POST /predict_cnn
Content-Type: multipart/form-data
```

- Corpo da Requisição: `file: imagem.png`

#### Rota da predição modelo linear

`POST /predict_mlp`
- Descrição: Recebe uma imagem de um dígito manuscrito e retorna a predição feita pelo modelo MLP.
- Parâmetros do Corpo da Requisição: file - Arquivo de imagem a ser enviado no formato multipart/form-data.
- Resposta:
```json
 {
    "digit_mlp": int,
    "inference_time_mlp": float
}
```
    `digit_mlp`: Dígito predito pelo modelo MLP.
    `inference_time_mlp`: Tempo de inferência do modelo MLP em segundos.

- Exemplo de Requisição:
```
POST /predict_mlp
Content-Type: multipart/form-data
```

- Corpo da Requisição: `file: imagem.png`

---
### Execução

1. Clone este repositório
```bash
git clone https://github.com/rafaelarojas/mnist.git
```

2. Instale todas as dependências
```bash
pip install -r requirements.txt
```

3. E certifique que algumas bibliotecas importantes estão instaladas
```bash
pip install flask tensorflow numpy os
```

4. Para executar, certifique-se que está dentro da pasta `mnist/src` e rode
```bash
python3 app.py
```