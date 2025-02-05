from flask import Flask, request, render_template
import joblib
import spacy
from elasticsearch import Elasticsearch

app = Flask(__name__)

try:
    classifier = joblib.load('text_classifier_model.pkl')
    vectorizer = joblib.load('tfidf_vectorizer.pkl')
    nlp = spacy.load("ner_fashion_model")
except Exception as e:
    print(f"Error loading models: {e}")

try:
    es_host = "https://localhost:9200"
    es_username = "elastic"
    es_password = "WLr9YJy_OSe=POPdH3d1"
    es = Elasticsearch(
        es_host,
        http_auth=(es_username, es_password),
        verify_certs=False
    )
except Exception as e:
    print(f"Error connecting to Elasticsearch: {e}")

index_name = "products"

RESULTS_PER_PAGE = 10

category_image_mapping = {
    'Bottom Wear': 'images/bottomwear.png',
    'Topwear': 'images/topwear.png',
    'Lingerie & Sleep Wear': 'images/lingerie.png',
    'Western': 'images/western.png',
    'Sports Wear': 'images/sportswear.png',
    'Indian Wear': 'images/indianwear.png',
    'Plus Size': 'images/plussize.png',
    'Inner Wear & Sleep Wear': 'images/sleepwear.png'
}

@app.route('/', methods=['GET', 'POST'])
def index():
    search_query = request.args.get('search', '')
    page = int(request.args.get('page', 1))
    offset = (page - 1) * RESULTS_PER_PAGE
    simple_search_results = []
    advanced_search_results = []
    identified_brand = None
    predicted_category = None
    simple_total_results = 0
    advanced_total_results = 0

    if search_query:
        simple_search_body = {
            "query": {
                "match": {
                    "description": search_query
                }
            },
            "from": offset,
            "size": RESULTS_PER_PAGE
        }
        simple_response = es.search(index=index_name, body=simple_search_body)
        simple_total_results = simple_response['hits']['total']['value']
        
        for hit in simple_response['hits']['hits']:
            product = hit['_source']
            product_category = product.get('category')
            product['image_source'] = category_image_mapping.get(product_category, 'images/indianwear.png')
            simple_search_results.append(product)

        X_new = vectorizer.transform([search_query])
        predicted_category = classifier.predict(X_new)[0]

        doc = nlp(search_query)
        for ent in doc.ents:
            if ent.label_ == 'BRAND':
                identified_brand = ent.text
                break

        must_clauses = [{"match": {"description": search_query}}]
        if identified_brand:
            must_clauses.append({"match": {"brand": identified_brand}})
        if predicted_category:
            must_clauses.append({"match": {"category": predicted_category}})
        advanced_search_body = {
            "query": {
                "bool": {
                    "must": must_clauses
                }
            },
            "from": offset,
            "size": RESULTS_PER_PAGE
        }
        advanced_response = es.search(index=index_name, body=advanced_search_body)
        advanced_total_results = advanced_response['hits']['total']['value']
        
        for hit in advanced_response['hits']['hits']:
            product = hit['_source']
            product_category = product.get('category')
            product['image_source'] = category_image_mapping.get(product_category, 'images/indianwear.png')
            advanced_search_results.append(product)

    return render_template('index.html', 
                           simple_search_results=simple_search_results, 
                           advanced_search_results=advanced_search_results, 
                           search_query=search_query,
                           identified_brand=identified_brand,
                           predicted_category=predicted_category,
                           simple_total_results=simple_total_results,
                           advanced_total_results=advanced_total_results,
                           page=page,
                           results_per_page=RESULTS_PER_PAGE)

if __name__ == '__main__':
    app.run(debug=True,port=5001)

