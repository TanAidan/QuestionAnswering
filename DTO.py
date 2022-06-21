from dataclasses import dataclass


@dataclass
class QAObject(object):
    question: str
    transcript: str
    mem_id: str
    score: float = 0.0

@dataclass
class QAList(object):
    qa_object_list: list[QAObject]