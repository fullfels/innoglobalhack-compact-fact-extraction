"""An explicit ML prompt view; corpus labels and source archives stay unchanged."""
from copy import deepcopy

OLD = "При отсутствии отдельного времени используй started_at; невосстановимое время оставь null."
NEW = ("Для утверждения о текущем состоянии без отдельной даты started_at задаёт момент наблюдения. "
       "Для прошлого события и будущего срока нужна дата, выводимая из разговора. "
       "Если точная историческая дата или срок не восстанавливаются, верни null; "
       "не заменяй неизвестную дату временем звонка.")


def effective_instruction(text):
    if text.count(OLD) > 1:
        raise ValueError("Unexpected repeated time fallback instruction")
    return text.replace(OLD, NEW)


def effective_example(row):
    result = deepcopy(row)
    result["prompt"][0]["content"] = effective_instruction(result["prompt"][0]["content"])
    return result
