# Se declaran las librerias a utilizar
import torch
import openai
import string
import warnings
from pysentimiento import create_analyzer
from transformers import AutoTokenizer, AutoModelForSequenceClassification
# Se deshabilitan los warnings
warnings.filterwarnings("ignore", category=FutureWarning)
# Se crea un diccionario de idiomas para poder consumir facilmente los modelos
# TO DO agregar mas idiomas
idiom = {"en": ["en_core_web_sm", 'english', 'wordnet', "en"]}
# Se crea la clase sentimentModel
class sentimentModel():
    # Se crea la funcion __init__ que instancia la clase
    def __init__(self, lenguaje = "en"):
        # Se inicializan los modelos de sentmiento que se utilizaran
        self.pysent_sent = create_analyzer(task="sentiment", lang=idiom[lenguaje][3])
        self.finbert_tokenizer = AutoTokenizer.from_pretrained("ProsusAI/finbert")
        self.finbert_model = AutoModelForSequenceClassification.from_pretrained("ProsusAI/finbert")
    # Se crea la funcion pysentmiento_analisis_text, que permitira consumir el modelo de pysentimiento
    def pysentimiento_analisis_text(self, text, proba = False):
        # Se inicializa el modelo de pysentimiento
        sentiment = self.pysent_sent.predict(text)
        # Si se habilita la bandera proba se retorna el diccionario con las probabilidades obtenidas
        if proba:
            return sentiment
        # Si la bandera proba esta deshabilitada se retorna la etiqueta de acuerdo a las probabilidades obtenidas
        else:
            # Se saca la etiqueta de la clase predicha 
            Key_max = max(zip(sentiment.probas.values(), sentiment.probas.keys()))[1]
            # Se asigna el valor de acuerdo a la etiqueta predicha 
            if Key_max == "NEG":
                return -1
            elif Key_max == "POS":
                return 1
            else:
                return 0
    # Se crea la funcion finbert_sentiment_analisis_text, que permite cosumir el modelo de finbert        
    def finbert_sentiment_analisis_text(self, text, proba = False):
        # Se crea una lista con los valores correspondientes a la pocison de las etiquetas 
        conversion = [1, -1, 0]
        # Se realizar una prediccion con el modelo findbert
        input = self.finbert_tokenizer(text, padding = True, truncation = True, return_tensors='pt')
        outputs = self.finbert_model(**input)
        # se aplica una funcion softmax para convertir la salida en probabilidades
        sentiment = torch.nn.functional.softmax(outputs.logits, dim=-1)
        # Se transforma los resultados de un tensor a una lista
        out = list(sentiment[0])
        # Se obtiene el valor maximo y la etiqueta que corresponde a ese valor maximo
        max_value = max(out)
        index = out.index(max_value)
        # Si se tiene activada la bandera de proba se retorna todo el tensor resultante, de lo contrario solo se 
        # retorna la etiqueta correspondiente
        if proba:
            return sentiment
        else:
            return conversion[index]
    # Se crea la funcion assembler_model, en la cual se pondera los resulatods de los modelos de pysentimiento y 
    # el de Finbert    
    def assembler_model(self, pysentimiento, finbert, weight):
        # Se pondera los resultados de los dos modelos baso un parametro de peso para obtener un nuevo resultado
        POS = (pysentimiento.probas["POS"])*weight + (finbert[0][0]*(1-weight))
        NEG = (pysentimiento.probas["NEG"])*weight + (finbert[0][1]*(1-weight))
        # Se retorna una tupla con los resultados
        return (POS, NEG)
    # Se crea la funcion assembler_model_predict que combina los modelos de Pysentimiento y Finbert
    def assembler_model_predict(self, text, weight = 0.9, tresh = 0.1):
        # Se inicaliza la salida en cero
        out = 0
        # Se realiza una prediccion con el modelo Pysentimiento y Findbert
        pysentimiento = self.pysentimiento_analisis_text(text, True)
        findbert = self.finbert_sentiment_analisis_text(text, True)
        # Se realiza una ponderacion de los resultados con el peso determinado
        POS, NEG = self.assembler_model(pysentimiento, findbert.detach().numpy(), weight)
        # Se asigna la etiqueta de acuerdo al resultado
        if NEG > POS:
            if NEG > tresh:
                out = "-1"
            else:
                out = "0"
        else:
            if POS > tresh:
                out = "1"
            else:
                out = "0"
        # Se retorna el resultado
        return out
    # Se crea la funcion chatgpt_predict que consumira el api de openia (ChatGPT)
    def chatgpt_predict(self, text, key, model_chat = "gpt-3.5-turbo"):
        # Se realiza un try en caso de que la api no responda adecuadamente
        try:
            # Se verifica que se ingrese una key
            if key == None:
                # En caso de no ingresar una key se retorna un None y se imprime un mensaje
                print("Falta ingresar la key para consumir la api de openIA")
                return None
            # En caso de que si ingrese una key se procede a consumir la api de openai
            else: 
                # Se añade la key al objecto de openai
                openai.api_key = key
                # Se limita el texto a 3000 tokens (palabras) por restriccion del modelo gpt 3.5, pero se puede
                # moficar hasta 8000 en caso de usar gpt 4
                text = " ".join(text.split()[:3000])
                # Se crea una frace inicial que permitira retornar solamente el sentimiento de la noticia
                sentence = "Tell me only what feeling between positive, negative and neutral has the following news, without justification"
                # Se Añade la frace al inicio del texto
                text = sentence + "\n" + text
                # Se consume la api de openai
                result = openai.ChatCompletion.create(
                    model = model_chat,
                    messages = [
                        {"role": "user", "content": text}
                    ]
                )
                # Se procesa el resultado para obtener unicamente el sentimiento 
                sentiment = result["choices"][0]["message"]["content"].lower()
                sentiment = sentiment.translate(str.maketrans('', '', string.punctuation))
                # Se asigna la etiqueta correspondiente al sentimiento
                predict = ""
                if  "neutral" in sentiment:
                    predict = "0"
                elif "positive" in sentiment:
                    predict = "1"
                elif "negative" in sentiment:
                    predict = "-1"
                else: 
                    # Se imprime un mensahe de error en caso de que no se retorne ninguna prediccion valida y se 
                    # se retorna None
                    print("Error de prediccion")
                    predict = None
        # En caso de algun error se imprime el error y se retorna None
        except:
            print("Error con la api de openAI")
            predict = None
        # Se retorna el resultado
        return predict




    