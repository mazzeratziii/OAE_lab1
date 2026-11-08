import argparse
import json
import logging
from pathlib import Path
from tempfile import TemporaryDirectory
from time import perf_counter

from langchain_core.exceptions import OutputParserException
from langchain_core.messages import AIMessage
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnableLambda

from catalog import initialize, search
from lab01 import BASIC, PARSER, PROMPT, TOOL, TOOL_PROMPT, ProductQuery, client, inputs, pipeline

CASES = [
    ("Найди ноутбуки дешевле 70 000 рублей", "laptop", 69999, "price_asc"),
    ("Покажи самые дорогие смартфоны", "smartphone", None, "price_desc"),
    ("Хочу посмотреть планшеты", "tablet", None, "price_asc"),
    ("Покажи ноутбуки до 50 тысяч", "laptop", 50000, "price_asc"),
    ("Найди холодильники", "other", None, "price_asc"),
    ("Хочу недорогой планшет", "tablet", None, "price_asc"),
    ("Смартфон или планшет около 30 тысяч", "smartphone", 30000, "price_asc"),
    ("Ноутбуки до 10 тысяч", "laptop", 10000, "price_asc"),
    ("Сначала дорогие планшеты до 30 тысяч", "tablet", 30000, "price_desc"),
    ("Смартфоны до 18 тысяч", "smartphone", 18000, "price_asc"),
]
DIALOGUES = [
    [("Покажи ноутбуки до 80 тысяч", "laptop", 80000, "price_asc"),
     ("А сначала дорогие?", "laptop", 80000, "price_desc"),
     ("А что есть среди смартфонов?", "smartphone", 80000, "price_desc")],
    [("Планшеты до 25 тысяч", "tablet", 25000, "price_asc"),
     ("Теперь до 70 тысяч", "tablet", 70000, "price_asc"),
     ("А без ограничения цены?", "tablet", None, "price_asc")],
]
BAD = [
    '{"category": @}',
    '{"category": "fridge"}',
    '{"category": "tablet", "max_price": -1}',
]


def expected(case):
    _, category, max_price, sort_by = case
    return ProductQuery(category=category, max_price=max_price, sort_by=sort_by)


def inspect(response, case):
    try:
        params = PARSER.invoke(response).model_dump()
    except OutputParserException:
        return {"params": None, "correct": False, "parse_error": True}
    checks = {key: value == params[key] for key, value in expected(case).model_dump().items()}
    return {"params": params, "field_checks": checks, "correct": all(checks.values())}


class ReplayToolModel:
    def __init__(self, response):
        self.response = response

    def bind_tools(self, _):
        return RunnableLambda(lambda _: self.response)


def offline():
    errors = []
    for raw in BAD:
        try:
            PARSER.invoke(raw)
        except OutputParserException as error:
            errors.append({"input": raw, "error": type(error).__name__,
                           "cause": type(error.__cause__).__name__})
        else:
            raise AssertionError("Невалидный ответ принят.")

    retries = []
    with TemporaryDirectory() as directory:
        db = Path(directory) / "catalog.sqlite3"
        initialize(db)
        fixtures = []
        for case in CASES:
            params = expected(case)
            products = search(params, db)
            prices = [row["price"] for row in products]
            assert all(row["category"] == params.category for row in products)
            assert params.max_price is None or all(price <= params.max_price for price in prices)
            assert prices == sorted(prices, reverse=params.sort_by == "price_desc")
            fixtures.append({"query": case[0], "params": params.model_dump(), "products": products})

        assert [len(row["products"]) for row in fixtures] == [3, 5, 5, 1, 0, 5, 2, 0, 3, 1]
        good = expected(CASES[0]).model_dump_json()
        for enabled, replies, wanted in [
            (False, [BAD[0], good], (1, 0)),
            (True, [BAD[0], good], (2, 1)),
            (True, BAD, (3, 0)),
        ]:
            calls = []
            answers = iter(replies)

            def generate(_):
                calls.append("llm")
                return AIMessage(content=next(answers))

            def lookup(params):
                calls.append("search")
                return search(params, db)

            answer = pipeline(RunnableLambda(generate), retries=enabled, lookup=lookup).invoke(
                inputs(CASES[0][0])
            )
            assert (calls.count("llm"), calls.count("search")) == wanted
            assert calls == ["llm"] * wanted[0] + ["search"] * wanted[1]
            assert ("products" in answer) == bool(wanted[1])
            retries.append({"retry": enabled, "calls": calls, "answer": answer})
        tool_cases = []
        for response, wanted in [
            (AIMessage(content=""), False),
            (AIMessage(content="", tool_calls=[{
                "name": TOOL.name, "args": expected(CASES[0]).model_dump(), "id": "valid",
            }]), True),
            (AIMessage(content="", tool_calls=[{
                "name": TOOL.name, "args": {"category": "tablet", "max_price": -1}, "id": "invalid",
            }]), False),
        ]:
            answer = pipeline(
                ReplayToolModel(response), retries=False, tool=True,
                lookup=lambda params: search(params, db),
            ).invoke(inputs(CASES[0][0]))
            assert ("products" in answer) == wanted
            tool_cases.append(answer)
    return {"parser_errors": errors, "sqlite_fixtures": fixtures, "retry": retries,
            "tool_cases": tool_cases}


def live():
    initialize()
    model = client()
    request = {"query": CASES[0][0], "language": "русский"}
    direct = model.invoke(BASIC.invoke(request))
    basic = BASIC | model | StrOutputParser()
    basic_text = basic.invoke(request)
    batch = basic.batch([dict(request, query=case[0]) for case in CASES[:3]])
    results = {"basic": {"llm_type": type(direct).__name__, "chain_type": type(basic_text).__name__,
                         "chain_text": basic_text, "batch": batch}}
    structured = PROMPT | model | PARSER
    chain = pipeline(model)
    results["structured"] = []
    for case in CASES[:7]:
        try:
            params = structured.invoke(inputs(case[0]))
            checks = {key: value == getattr(params, key)
                      for key, value in expected(case).model_dump().items()}
            row = {"json": params.model_dump_json(), "field_checks": checks,
                   "correct": all(checks.values())}
        except OutputParserException:
            row = {"parse_error": True, "correct": False}
        results["structured"].append({"query": case[0], **row})
    results["search"] = []
    for index, case in enumerate(CASES):
        answer = chain.invoke(inputs(case[0]))
        row = {"query": case[0], **answer, "correct": (
            answer.get("params") == expected(case).model_dump()
            and answer.get("products") == search(expected(case))
        )}
        if index < 3:
            question = case[0] + ". Какие товары есть в каталоге?"
            row["without_catalog"] = model.invoke(question).content
        results["search"].append(row)

    results["thinking"] = []
    for enabled, chosen in [(False, model), (True, client(thinking=True))]:
        for case in CASES[5:7]:
            prompt = PROMPT.invoke(inputs(case[0]))
            start = perf_counter()
            response = chosen.invoke(prompt)
            elapsed = perf_counter() - start
            usage = response.usage_metadata or {}
            results["thinking"].append({
                "query": case[0], "enabled": enabled, "seconds": round(elapsed, 3),
                "output_tokens": usage.get("output_tokens"), "content": response.content,
                "output_token_details": usage.get("output_token_details"),
                "reasoning_fields": [key for key in response.additional_kwargs if "reason" in key],
                **inspect(response, case),
            })

    bound = model.bind_tools([TOOL])
    results["tools"] = []
    for case in CASES[:7]:
        response = bound.invoke(TOOL_PROMPT.invoke(inputs(case[0])))
        answer = pipeline(ReplayToolModel(response), retries=False, tool=True).invoke(
            inputs(case[0])
        )
        results["tools"].append({"query": case[0], "content": response.content,
                                 "tool_calls": response.tool_calls, **answer,
                                 "correct": answer.get("params") == expected(case).model_dump()})

    results["dialogues"] = []
    for number, dialogue in enumerate(DIALOGUES, 1):
        for mode in ("state", "history"):
            state, history = "", []
            for case in dialogue:
                context = state if mode == "state" else json.dumps(history, ensure_ascii=False)
                response = model.invoke(PROMPT.invoke(inputs(case[0], mode, context)))
                row = inspect(response, case)
                if row["params"] is not None:
                    state = ProductQuery.model_validate(row["params"]).model_dump_json()
                history.append(case[0])
                results["dialogues"].append({
                    "dialogue": number, "mode": mode, "query": case[0],
                    "input_tokens": (response.usage_metadata or {}).get("input_tokens"), **row,
                })
    return results


def main():
    cli = argparse.ArgumentParser()
    cli.add_argument("--live", action="store_true")
    cli.add_argument("--out", type=Path, default=Path("results.json"))
    args = cli.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    logging.getLogger("httpx").setLevel(logging.WARNING)
    try:
        data = live() if args.live else offline()
    except Exception as error:
        cli.exit(1, f"Сценарий завершился ошибкой ({type(error).__name__}).\n")
    args.out.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
    print(args.out)


if __name__ == "__main__":
    main()


