from copy import deepcopy

from decap.schemas.results import AnswerVersion


def snapshot(answer: AnswerVersion) -> AnswerVersion:
    return deepcopy(answer)


def rollback(saved: AnswerVersion) -> AnswerVersion:
    return deepcopy(saved)

