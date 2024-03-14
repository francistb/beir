from .. import BaseSearch
import base64, zlib, requests
import tqdm
from typing import List, Dict


def chunks(lst, n):
    """Yield successive n-sized chunks from lst."""
    for i in range(0, len(lst), n):
        yield lst[i:i + n]


class CoveoSearch(BaseSearch):
    def __init__(self, organization_id: str, source: str, api_key: str, search_api_key: str, environment: str = "dev",
                 mappings: Dict[str, str] = {"title": "title", "body": "text"}, batch_size: int = 5000
                 ):
        self.initialized = False
        self.organization = organization_id
        self.source = source
        self.api_key = api_key
        self.search_api_key = search_api_key
        self.environment = environment
        self.mappings = mappings
        self.batch_size = batch_size


    def search(self, corpus: Dict[str, Dict[str, str]], queries: Dict[str, str], top_k: int, *args, **kwargs) -> Dict[str, Dict[str, float]]:
        # Index the corpus within elastic-search
        # False, if the corpus has been already indexed
        if self.initialized:
            self.index(corpus)
            self.initialized = True

        coveo_session = requests.Session()
        coveo_session.headers.update({'Authorization': f'Bearer {self.search_api_key}'})
        search_endpoint = f'https://platform{self.environment}.cloud.coveo.com/rest/search/v3'

        # retrieve results from BM25
        results = {}
        for query_id, query in tqdm.tqdm(queries.items()):
            payload = {'q': ' OR '.join(query.replace('$', ' ').split(' ')), 'numberOfResults': 1000}
            r = coveo_session.post(search_endpoint, json=payload)
            if not r.ok:
                print(r.text)
                continue
            scores = {}
            hits = r.json()['results']
            for hit in hits:
                corpus_id = hit['uniqueId'].split('/')[-1]
                score = float(hit['score'])
                scores[corpus_id] = score
            results[query_id] = scores

        return results


    def push_documents(self, batch):
        PUSHAPI_ENDPOINT = f'https://api{self.environment}.cloud.coveo.com/'
        ORGANIZATION_ID = self.organization
        SOURCE_ID = self.source

        coveo_session = requests.Session()
        coveo_session.headers.update({'Authorization': f'Bearer {self.api_key}'})

        file_endpoint = f'{PUSHAPI_ENDPOINT}/push/v1/organizations/{ORGANIZATION_ID}/files'
        file_answer = coveo_session.post(file_endpoint).json()
        upload_uri = file_answer['uploadUri']
        file_id = file_answer['fileId']
        headers = file_answer['requiredHeaders']

        payload = {'addOrUpdate': batch}
        r = requests.put(upload_uri, json=payload, headers=headers)
        if not r.ok:
            print(r.text)

        pushapi_endpoint = f'{PUSHAPI_ENDPOINT}/push/v1/organizations/{ORGANIZATION_ID}/sources/{SOURCE_ID}/documents/batch?fileId={file_id}'
        r = coveo_session.put(pushapi_endpoint)
        if not r.ok:
            print(r.text)


    def prepare_document(self, _id, document):
        document['documentId'] = f'corpus://{_id}'
        document['compressedBinaryData'] = base64.encodebytes(
            zlib.compress(document[self.mappings['body']].encode())).decode()
        document['compressionType'] = 'ZLIB'
        document['fileExtension'] = ".txt"
        del document[self.mappings['body']]
        return document


    def index(self, corpus: Dict[str, Dict[str, str]]):
        progress = tqdm.tqdm(unit="docs", total=len(corpus))
        # dictionary structure = {_id: {title_key: title, text_key: text}}

        for documents in chunks(list(corpus.items()), self.batch_size):
            docs = [self.prepare_document(_id, doc) for (_id, doc) in documents]
            self.push_documents(docs)
            progress.update(len(docs))
