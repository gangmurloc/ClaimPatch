from decap.schemas.results import AnswerVersion


def next_version(answer: AnswerVersion) -> int:
    return answer.version + 1

