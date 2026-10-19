from roi_swarm.graph import normalize_plan
from roi_swarm.schemas import PlannedTask


def test_normalize_plan_never_pads_with_generic_copies_of_request() -> None:
    tasks = [
        PlannedTask(id="a", objective="Собрать цены"),
        PlannedTask(id="b", objective="Сравнить"),
    ]

    result = normalize_plan(tasks, maximum=70)

    assert [t.objective for t in result] == ["Собрать цены", "Сравнить"]
    assert all("aspect" not in t.objective for t in result)


def test_normalize_plan_drops_duplicate_and_empty_objectives() -> None:
    tasks = [
        PlannedTask(id="a", objective="Проверить  лимиты API"),
        PlannedTask(id="b", objective="проверить лимиты api."),
        PlannedTask(id="c", objective="   "),
        PlannedTask(id="a", objective="Оценить стоимость"),
    ]

    result = normalize_plan(tasks, maximum=70)

    assert [t.objective for t in result] == ["Проверить  лимиты API", "Оценить стоимость"]
    assert len({t.id for t in result}) == 2


def test_normalize_plan_caps_at_maximum() -> None:
    tasks = [PlannedTask(id=f"t{i}", objective=f"часть {i}") for i in range(250)]

    assert len(normalize_plan(tasks, maximum=200)) == 200
