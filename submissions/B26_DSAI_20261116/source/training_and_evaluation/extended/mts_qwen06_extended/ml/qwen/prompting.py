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
    result = text.replace(OLD, NEW)
    extra = ' Для проверяемой относительной даты можно заполнить source.quote точной цитатой исходной реплики; span оставляй null. Если надёжной дословной цитаты нет, quote=null. speaker копируй из метаданных выбранного source.turn_idx, не из предположений о роли человека.'
    return result if extra in result else result + extra


def effective_example(row):
    result = deepcopy(row)
    result["prompt"][0]["content"] = effective_instruction(result["prompt"][0]["content"])
    return result
