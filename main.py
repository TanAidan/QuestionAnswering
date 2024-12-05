import csv
import dataclasses

from connections.elastic_query import ElasticSearchConnection, ElasticQuery, memsnippet_idx_current
from elasticsearch_dsl import Search, Q, A, MultiSearch, DateRange

from DTO import QAObject


def main():
    # fetch list of identity ids from your database to train on
    identity_id_list = {}

    for identity_id in identity_id_list:
        es_helper = elasticSearchHelper(identity_id)

        file = "qg_" + str(identity_id) + "_" + identity_id_list[identity_id] + ".csv"
        with open(file, mode='w') as csv_file:
            fieldnames = ['mem_id', 'transcript', 'question', 'score']
            writer = csv.DictWriter(csv_file, fieldnames=fieldnames)

            writer.writeheader()
            for question in es_helper:
                question_dict = dataclasses.asdict(question)
                writer.writerow(question_dict)


def elasticSearchHelper(identity_id):
    try:
        q = ElasticMemblockQuery(identity_id)
        q.question_query()
        es_connection = ElasticSearchConnection()
        es_response = es_connection.send(q.search)
    except Exception as e:
        raise e

    try:
        memblock_questions = ElasticQuestionsResponse(es_response).get_results()
    except Exception as e:
        raise e
    return memblock_questions


class ElasticQuestionsResponse(object):

    def __init__(self, es_response):
        self.es_response = es_response
        self.hits = self.es_response.hits

    def get_results(self):
        question_list = []
        question_str_list = []
        for hit in self.hits:
            mem_id = hit.memdata.mem_id
            questions = hit.predicted_fields.question_generation.display
            transcript = hit.text_field.transcribed_text[0].name
            for question in questions:
                if question.question not in question_str_list:
                    question_list.append(QAObject(question=question.question, transcript=transcript, mem_id=mem_id))
                    question_str_list.append(question.question)
        return question_list


class ElasticMemblockQuery(ElasticQuery):
    source_include = ["predicted_fields.question_generation", "text_field.transcribed_text", "memdata.mem_id"]

    def __init__(self, identity_id, input=None):
        super().__init__()
        self.search = Search(index=memsnippet_idx_current + "_" + str(identity_id))
        self.identity_id = identity_id
        print("identity_id_ElasticMemblockQuery", identity_id)
        self.input = input

        self.min_doc_count = 1  # request ES to return terms aggs (facets) only when > min_doc_count
        self.terms_size = 100  # max number of facets to be returned

    def question_query(self):
        nested_path = "predicted_fields.question_generation"
        self.search = self.search.query("nested", path=nested_path, query=Q("exists", **{"field": nested_path}))
        self.set_size(10000)
        self.set_match("memdata.type", "memblock")
        self.set_source(source_list=self.source_include)
        self.set_sort("memdata.time_start", "desc")


if __name__ == '__main__':
    main()
