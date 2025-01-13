# Se declaran las librerias a utilizar
import time
import warnings
import pandas as pd
from tqdm import tqdm
from news_scraping.scraping import scraping
from sentiment_model.sentimentModel import sentimentModel
# Se deshabilitan los warnings
warnings.filterwarnings("ignore", category=FutureWarning)
# Se crea la clase newsAnalisis
class newsAnalisis():
    # Se crea la funcion __init__ que instancia la clase
    def __init__(self, lenguaje = "en"):
        # Se instancia la clase scraping y la clase sentimentModel
        self.scraping = scraping()
        self.sentimentModel = sentimentModel(lenguaje=lenguaje)
    # Se crea la funcion de busqueda
    def search(self, key, search_list, end_date, duration, n_pages = 1, type_time = "D", lenguage = "en", 
               region = "US"):
        # Se realiza el scraping en google news
        news = self.scraping.GoogleNews(search_list, end_date, duration, n_pages, type_time, lenguage, region)
        # Se imprime la cantidad de noticias encontradas
        print("Se encontraron " + str(len(news)) + " noticias")
        # Se inicializa una lista vacia en donde se almacenaran los resultados
        result = []
        # Se recorre las noticias obtenidas
        for i in tqdm(range(len(news))):
            # Se realiza una prediccion con el modelo de sentimientos que se construyo
            assembler_model_predict = self.sentimentModel.assembler_model_predict(news.iloc[i]["text"])
            # Se realiza una prediccion con chatgpt
            chatgpt_predict = self.sentimentModel.chatgpt_predict(news.iloc[i]["text"], key)
            # Se recopilan todos los resultados en una lista
            result.append([news.iloc[i]["search"], news.iloc[i]["Year"], news.iloc[i]["Month"], news.iloc[i]["Day"], 
                           news.iloc[i]["date"],  news.iloc[i]["media"], news.iloc[i]["description"], 
                           news.iloc[i]["title"], news.iloc[i]["link"], news.iloc[i]["text"], 
                           news.iloc[i]["word_count"], assembler_model_predict, chatgpt_predict])
            # Se agrega un timeout para no bloquear la api de openai (Ya que se usa la version de prueba)
            time.sleep(10)
        # Se crea un Dataframe a partir de la lista previamente generada    
        result = pd.DataFrame(result, columns=["search", "Year", "Month", "Day", "date", "media", "description", 
                                               "title", "link", "text", "word_count", "assembler_model_predict", 
                                               "chatgpt_predict"])
        # Se retorna el dataframe obtenido
        return result
            

        