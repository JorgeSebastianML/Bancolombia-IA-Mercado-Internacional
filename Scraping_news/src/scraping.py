# Importacion de librerias
import os
import re
import time
import json
import random
import datetime
import warnings
import pandas as pd
from tqdm import tqdm
import requests as req
from threading import Thread
from bs4 import BeautifulSoup
from .New_GoogleNews import New_GoogleNews
from dateutil.relativedelta import relativedelta
from transformers import TextClassificationPipeline, AutoModelForSequenceClassification, AutoTokenizer

# Se deshabilitan los warnings
warnings.filterwarnings("ignore", category=FutureWarning)

convert_dates = {"an": "Jan", 
                 "Feb": "Feb", 
                 "ar": "Mar", 
                 "pr": "Apr",
                 "May": "May",
                 "un": "Jun",
                 "Jul": "Jul",
                 "Aug": "Aug", 
                 "Sep": "Sep",
                 "ct": "Oct", 
                 "Nov": "Nov",
                 "Dec": "Dec"}
# Se crea una lista de proxies para variar el punto de origen del scraping
proxies = ['121.129.127.209:80', "173.192.21.89:80", "51.68.207.81:80",  
            "31.220.54.116:80", "104.16.241.204:80", "101.231.66.130:80", 
            "104.16.105.2:80", "104.18.254.76:80", "137.110.161.153:80", 
            "203.24.108.179:80"]
# Se lee una lista de navegadores para utilizar
navegadores = pd.read_csv("include/user_agent_list.csv")
# Se carga el modelo y el tokenizador para reconocer la publicidad
advertising_model = AutoModelForSequenceClassification.from_pretrained('morenolq/spotify-podcast-advertising-classification')
advertising_tokenizer = AutoTokenizer.from_pretrained('morenolq/spotify-podcast-advertising-classification')

class treah_scraping(Thread):
    # constructor
    def __init__(self, url, filtrado = True):
        # execute the base constructor
        Thread.__init__(self)
        # set a default values
        self.results = None
        # Se guarda el url como variable de la clase
        self.url = url
        # Se guarda la bandera filtrado como una vraible de la clase
        self.filtrado = filtrado

    # Funcion para limpiar caracteres expeciales del texto
    def Process_symbols_text(self, text):
        text = text.replace("\n", " ")
        text = text.replace("\t", " ")
        text = text.replace("\xa0", " ")
        text = re.sub(r"(@\[A-Za-z0-9]+)|([^0-9A-Za-z \t])|(\w+:\/\/\S+)|^rt|http.+?", "", text)
        text = " ".join(text.split())
        return text
    
    # Funcion para eliminar palabras muy largas del texto
    def clean_words(self, text, long = 13): 
        out = []
        for word in text.split():
            if len(word) <= long:
                out.append(word)
        return " ".join(out)
    
    # Funcion para limpiar patrones de publicidad identificados del texto
    def clean_patrons(self, text, path = "include/patrons.json"):
        f = open(path)
        dict_patrons = json.load(f)
        for key in list(dict_patrons.keys()):
            patron = dict_patrons[key]
            out = text.split(patron)
            text = " ".join(out)
        f.close()
        return text
    
    # Funcion para separar el texto en porciones, para un posterior procesamiento
    def separate_tex(self, text, n_tokens = 100):
        initial_text = text.split(". ")
        select_text = []
        for text in initial_text:
            text = " ".join(text.split())
            if len(text.split()) > 5:
                if len(text.split()) > n_tokens:
                    texto = text.split()
                    parts = [texto[i:i+n_tokens] for i in range(0, len(texto), n_tokens)]
                    for part in parts:
                        select_text.append(" ".join(part))
                else:
                    select_text.append(text)
        return select_text
    
    # modelo para limpiar publicidad (Solo para ingles)
    def advertising_clean(self, text):
        pipe = TextClassificationPipeline(model=advertising_model, 
                                        tokenizer=advertising_tokenizer)
        prediction = pipe(text, return_all_scores=True)
        if prediction[0][1]["score"] <= prediction[0][0]["score"]:
            return text
        else:
            return None
 
    # function executed in a new thread
    def run(self):
        # Se selecciona una proxy aleatoriamente
        proxy = random.choice(proxies)
        # Se selecciona un navegador de forma aleatoria
        headers = {'User-Agent': navegadores.sample(1)["user-agent"].values[0]}
        # Se realiza la peticion por medio de request
        response=req.get(self.url,headers=headers, proxies={"http": proxy, "https": proxy})
        # En caso se que retorne un codigo 403 se retorna un error 
        if response == 403 or response == 449:
            print("-------------------------------------------------")
            print("------------- Demasiadas peticiones -------------")
            print("-------------------------------------------------")
            return "Error"
        # Por mecio de BeautifulSoup se extrae el contendio de la pagina 
        soup=BeautifulSoup(response.content,'lxml')
        # si el filtro esta activado se limpia el texto para reducir el ruido en ella
        if self.filtrado:
            # Limpieza de simbolos y links
            text = self.Process_symbols_text(soup.get_text())
            # Limpieza de palabras muy largas, (Generalmente son ruido)
            text = self.clean_words(text)
            # Limpieza de patrones
            text = self.clean_patrons(text.lower())
            # Se se para el texto y se revisa que parte es publicidad
            separate_text = self.separate_tex(text)
            clean_text = []
            for part in separate_text:
                temp = self.advertising_clean(part)
                if temp != None:
                    clean_text.append(temp)
            text = " ".join(clean_text)
            self.results = text
        else: 
            self.results = soup.get_text()

        return (self.results, soup.get_text())
        

# Se crea la clase scraping
class scraping:
    # Funcion de inicilizacion de la clase
    def __init__(self):
        # Se crea una lista de proxies para variar el punto de origen del scraping
        self.proxies = ['121.129.127.209:80', "173.192.21.89:80", "51.68.207.81:80",  
                    "31.220.54.116:80", "104.16.241.204:80", "101.231.66.130:80", 
                    "104.16.105.2:80", "104.18.254.76:80", "137.110.161.153:80", 
                    "203.24.108.179:80"]
        # Se lee una lista de navegadores para utilizar
        self.navegadores = pd.read_csv("include/user_agent_list.csv")
        # Se carga el modelo y el tokenizador para reconocer la publicidad
        self.advertising_model = AutoModelForSequenceClassification.from_pretrained('morenolq/spotify-podcast-advertising-classification')
        self.advertising_tokenizer = AutoTokenizer.from_pretrained('morenolq/spotify-podcast-advertising-classification')

    # Funcion para verificar los formatos de los parametros ingresados
    def Format_Verification(self, date, duration, n_pages, type_time, lenguage, region):
        # Se verifica los formatos de los parametros ingresados
        assert isinstance(date, str), "La fecha debe ser ingresada como string"
        assert isinstance(duration, int), "La duraccion debe ser ingresada como int"
        assert isinstance(n_pages, int), "El numero de paginas debe ser ingresada como int"
        assert isinstance(type_time, str), "El timepo debe ser ingresada como string"
        try:
            datetime.datetime.strptime(date,"%Y-%m-%d")
        except ValueError as err:
            print(err)
        if type_time != "D" and type_time != "MS" and type_time != "W":
            assert type_time == "null", "Valor type_time invalido, seleccionar entre D (Diario), W (Semanal), MS (Mensual)"

        assert isinstance(lenguage, str), "El lenguaje debe ser ingresada como string"
        assert isinstance(region, str), "La region debe ser ingresada como string"

    # Funcion para limpiar caracteres expeciales del texto
    def Process_symbols_text(self, text):
        text = text.replace("\n", " ")
        text = text.replace("\t", " ")
        text = text.replace("\xa0", " ")
        text = re.sub(r"(@\[A-Za-z0-9]+)|([^0-9A-Za-z \t])|(\w+:\/\/\S+)|^rt|http.+?", "", text)
        text = " ".join(text.split())
        return text
    
    # Funcion para eliminar palabras muy largas del texto
    def clean_words(self, text, long = 13): 
        out = []
        for word in text.split():
            if len(word) <= long:
                out.append(word)
        return " ".join(out)
    
    # Funcion para limpiar patrones de publicidad identificados del texto
    def clean_patrons(self, text, path = "include/patrons.json"):
        f = open(path)
        dict_patrons = json.load(f)
        for key in list(dict_patrons.keys()):
            patron = dict_patrons[key]
            out = text.split(patron)
            text = " ".join(out)
        f.close()
        return text
    
    # Funcion para separar el texto en porciones, para un posterior procesamiento
    def separate_tex(self, text, n_tokens = 100):
        initial_text = text.split(". ")
        select_text = []
        for text in initial_text:
            text = " ".join(text.split())
            if len(text.split()) > 5:
                if len(text.split()) > n_tokens:
                    texto = text.split()
                    parts = [texto[i:i+n_tokens] for i in range(0, len(texto), n_tokens)]
                    for part in parts:
                        select_text.append(" ".join(part))
                else:
                    select_text.append(text)
        return select_text
    
    # modelo para limpiar publicidad (Solo para ingles)
    def advertising_clean(self, text):
        pipe = TextClassificationPipeline(model=self.advertising_model, 
                                        tokenizer=self.advertising_tokenizer)
        prediction = pipe(text, return_all_scores=True)
        if prediction[0][1]["score"] <= prediction[0][0]["score"]:
            return text
        else:
            return None

    # funcion que permite realizar un scraping a una pagina web por medio de su link
    def scraping_web(self, url, filtrado = True):
        # Se selecciona una proxy aleatoriamente
        proxy = random.choice(self.proxies)
        # Se selecciona un navegador de forma aleatoria
        headers = {'User-Agent': self.navegadores.sample(1)["user-agent"].values[0]}
        # Se realiza la peticion por medio de request
        response=req.get(url,headers=headers, proxies={"http": proxy, "https": proxy})
        # En caso se que retorne un codigo 403 se retorna un error 
        if response == 403:
            return "Error"
        # Por mecio de BeautifulSoup se extrae el contendio de la pagina 
        soup=BeautifulSoup(response.content,'lxml')
        # si el filtro esta activado se limpia el texto para reducir el ruido en ella
        if filtrado:
            # Limpieza de simbolos y links
            text = self.Process_symbols_text(soup.get_text())
            # Limpieza de palabras muy largas, (Generalmente son ruido)
            text = self.clean_words(text)
            # Limpieza de patrones
            text = self.clean_patrons(text.lower())
            # Se se para el texto y se revisa que parte es publicidad
            separate_text = self.separate_tex(text)
            clean_text = []
            for part in separate_text:
                temp = self.advertising_clean(part)
                if temp != None:
                    clean_text.append(temp)
            text = " ".join(clean_text)
            return text
        else: 
            return soup.get_text()

    # Funcion de scraping para google News
    def GoogleNews(self, search_list, end_date, duration, n_pages = 1, type_time = "D", lenguage = "en", 
                   region = "US"):
        # Se verifican los formatos de los parametros ingresados
        self.Format_Verification(end_date, duration, n_pages, type_time, lenguage, region)
        # Se verifica si search list es una path o una lista
        if type(search_list) == str:
            assert os.path.isfile(search_list), "El archivo " + search_list + " No existe"
            List_Topics = list(pd.read_excel(search_list, engine="openpyxl")["Names"])
        elif type(search_list) == list:
            List_Topics = search_list
        else:
            assert isinstance(search_list, str), "La lista de compañias debe ser una lista o un archivo excel"
        
        # Se tranforma la fecha en datetime
        end = datetime.date.fromisoformat(end_date)
        # Se calcula el delta de tiempo de la busqueda
        if type_time == "D":
            start = end - relativedelta(days=duration)
        elif type_time == "W":
            start = end - relativedelta(weeks=duration) 
        else: 
            start = end - relativedelta(months=duration)
        # Se crea una lista con las fechas en las que se realizara la busqueda
        dates_in_range = [x.split(' ') for x in pd.date_range(start, end, freq=type_time).strftime("%m/%d/%Y").tolist()]
        
        if duration == 0:
            dates_in_range = [[end.strftime("%m/%d/%Y")], [end.strftime("%m/%d/%Y")]]
        
        print("------------------------------------------------------------------")
        print("----------------------- Iniciando --------------------------------")
        print("------------------------------------------------------------------")
        # Se imprime las fechas en las que se realizara la busqueda
        print("Fecha de inicio: " + dates_in_range[0][0])
        print("Fecha final: " + dates_in_range[-1][0])
        # Se crea una lista vacia en donde se almacenaran los resultados de la busqueda
        articles = []
        # Se recorre la lista de fechas
        for date in tqdm(range(len(dates_in_range) - 1)):
            # Se estrae las fechas en las que se realizara la busqueda
            start_date = dates_in_range[date][0]
            end_date = dates_in_range[date + 1][0]
            # Se recorre la lista de topicos que se desean buscar
            for t in List_Topics: 
                # Se instancia el objecto de New_GoogleNews
                googlenews = New_GoogleNews(lang=lenguage, region=region, start=start_date,end=end_date,
                                            header=self.navegadores.sample(1)["user-agent"].values[0], 
                                            proxy = random.choice(self.proxies))
                # Se setea el encoder al utilizar
                googlenews.set_encode('utf-8')
                # Se realiza la busqueda en google news
                #googlenews.search(t)
                # Se realiza un recorrido sobre las paginas de resultados
                for i in range(n_pages):
                    #googlenews.get_news(t)
                    googlenews.search(t)
                    #googlenews.page_at(i)
                    # Se organizan los resultados           
                    documents = googlenews.results(sort=True)
                    try:
                        count_1 = 0
                        
                        for doc in documents:
                            # Se organiza la informacion obtenida en un diccionario
                            filteredDoc = {}
                            filteredDoc["company"] = t
                            try:
                                Month = convert_dates[doc['date'].split()[0]]
                                fecha = Month + " " + doc['date'].split()[1] + " " + doc['date'].split()[2]
                                fecha = datetime.datetime.strptime(fecha, '%b %d, %Y')
                                filteredDoc["Year"] = fecha.year
                                filteredDoc["Month"] = fecha.month
                                filteredDoc["Day"] = fecha.day
                                filteredDoc['date'] = fecha.strftime("%Y/%m/%d") 
                            except:
                                fecha = datetime.datetime.strptime(end_date, '%m/%d/%Y')
                                filteredDoc["Year"] = fecha.year
                                filteredDoc["Month"] = fecha.month
                                filteredDoc["Day"] = fecha.day
                                filteredDoc['date'] = fecha.strftime("%Y/%m/%d") 
                            if doc['media'] != None:
                                filteredDoc['media'] = doc['media']
                                filteredDoc['description'] = doc['desc']
                            else:
                                filteredDoc['media'] = doc['desc']
                                filteredDoc['description'] = ""
                            filteredDoc['title'] = doc['title']
                            filteredDoc['link'] = doc['link']
                            if doc['link'][0:11] == 'news.google':
                                url="https://" + doc['link']
                                # Se utiliza la funcion scraping_web para retronar la url real de la 
                                # noticia
                                result = self.scraping_web(url, filtrado = False)
                                link = result.split(" ")[-1][0:-5]
                            else: 
                                link = doc["link"]
                            bandera = True
                            cont = 0
                            while bandera:
                                if bandera:
                                    # Se utiliza la funcion scraping_web para retornar informacion de la
                                    # pagina de interes
                                    #result = self.scraping_web(link)
                                    thread = treah_scraping(link, filtrado = False)
                                    # start the thread
                                    thread.start()
                                    # wait for the thread to finish
                                    thread.join(30)
                                    results = thread.results
                                    result_1, result_2 = results
                                    # En caso de que se obtenga correctamente la informacion se termina 
                                    # el ciclo
                                    if result_2 != "Error":
                                        bandera = False
                                    # Se incrementa un contador y se realiza un sleep de 6 segundos para
                                    # no saturar la paginas con peticiones
                                    cont += 1
                                    time.sleep(random.randint(1, 6))
                                    # En caso de requerir mas intentos que proxies disponibles se genera
                                    # un error 
                                    if cont >= len(self.proxies):
                                        assert isinstance(result_2, int), "Demasiadas peticiones, esperar al menos 30 min"
                            # Se guarda el texto obtenido en la busqueda y su conteo de palabras
                            if result_2 != None:
                                filteredDoc["Text"] = result_2
                                filteredDoc["Crude_Text"] = result_1
                                filteredDoc["word_count"] = len(result_2.split())
                                # Se guarda el resultado en una lista
                                if filteredDoc["word_count"] > 15:
                                    articles.append(filteredDoc)
                            if count_1 >= 20:
                                break
                            else: 
                                count_1 += 1
                    except:
                        pass
        print("-------------------------------")
        print("------ creando dataframe ------")
        print("-------------------------------")
        # Se crea un dataframe con los resultados y se retorna
        articles = pd.DataFrame(articles)
        return articles