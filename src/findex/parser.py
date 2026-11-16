from dataclasses import dataclass
import re
from src.findex.core import tokenize
from src.findex.index import Index


class QuerySyntaxError(ValueError):
    """Ошибка синтаксиса поискового запроса."""


class QueryNode:
    """Базовый узел AST-дерева запроса с поддержкой операторов &, |, ~."""

    def __and__(self, other: "QueryNode") -> "And":
        return And(self, other)

    def __or__(self, other: "QueryNode") -> "Or":
        return Or(self, other)

    def __invert__(self) -> "Not":
        return Not(self)

    def evaluate(self, index: Index) -> set[int]:
        raise NotImplementedError

    def positive_terms(self) -> list[str]:
        """Возвращает термины (кроме отрицаний NOT) для подсчёта релевантности."""
        return []

    def phrases(self) -> list[tuple[str, ...]]:
        """Возвращает точные фразы из запроса (для подсветки сниппетов)."""
        return []


@dataclass(frozen=True, slots=True)
class Term(QueryNode):
    term: str

    def evaluate(self, index: Index) -> set[int]:
        if self.term not in index:
            return set()
        return {p.doc_id for p in index[self.term]}

    def positive_terms(self) -> list[str]:
        return [self.term]


@dataclass(frozen=True, slots=True)
class Phrase(QueryNode):
    terms: tuple[str, ...]

    def evaluate(self, index: Index) -> set[int]:
        if not self.terms:
            return set()
        if not index.store_positions:
            raise ValueError("Фразовый поиск требует индекс с сохранёнными позициями слов.")
        if len(self.terms) == 1:
            return Term(self.terms[0]).evaluate(index)

        postings_per_term: list[dict[int, set[int]]] = []
        for term in self.terms:
            if term not in index:
                return set()
            postings_per_term.append({p.doc_id: set(p.positions) for p in index[term]})

        common_docs = set.intersection(*(set(d.keys()) for d in postings_per_term))
        matched_docs: set[int] = set()

        for doc_id in common_docs:
            first_positions = postings_per_term[0][doc_id]
            for start_pos in first_positions:
                if all(
                    (start_pos + offset) in postings_per_term[offset][doc_id]
                    for offset in range(1, len(self.terms))
                ):
                    matched_docs.add(doc_id)
                    break

        return matched_docs

    def positive_terms(self) -> list[str]:
        return list(self.terms)

    def phrases(self) -> list[tuple[str, ...]]:
        return [self.terms]


@dataclass(frozen=True, slots=True)
class And(QueryNode):
    left: QueryNode
    right: QueryNode

    def evaluate(self, index: Index) -> set[int]:
        return self.left.evaluate(index) & self.right.evaluate(index)

    def positive_terms(self) -> list[str]:
        return self.left.positive_terms() + self.right.positive_terms()

    def phrases(self) -> list[tuple[str, ...]]:
        return self.left.phrases() + self.right.phrases()


@dataclass(frozen=True, slots=True)
class Or(QueryNode):
    left: QueryNode
    right: QueryNode

    def evaluate(self, index: Index) -> set[int]:
        return self.left.evaluate(index) | self.right.evaluate(index)

    def positive_terms(self) -> list[str]:
        return self.left.positive_terms() + self.right.positive_terms()

    def phrases(self) -> list[tuple[str, ...]]:
        return self.left.phrases() + self.right.phrases()


@dataclass(frozen=True, slots=True)
class Not(QueryNode):
    operand: QueryNode

    def evaluate(self, index: Index) -> set[int]:
        return index.all_doc_ids() - self.operand.evaluate(index)


@dataclass(frozen=True, slots=True)
class _QToken:
    kind: str  # WORD, PHRASE, AND, OR, NOT, LPAREN, RPAREN
    value: str


_TOKEN_RE = re.compile(
    r'\s*(?:'
    r'"([^"]*)"'          # группа 1: фраза в кавычках
    r'|(\()'              # группа 2: (
    r'|(\))'              # группа 3: )
    r'|(&|\|\||\| |~)'    # группа 4: символьные операторы
    r"|([\w'-]+)"         # группа 5: слово или AND/OR/NOT
    r')'
)


def _lex_query(query: str) -> list[_QToken]:
    tokens: list[_QToken] = []
    pos = 0
    while pos < len(query):
        if query[pos].isspace():
            pos += 1
            continue
        match = _TOKEN_RE.match(query, pos)
        if not match:
            raise QuerySyntaxError(f"Неожиданный символ в запросе на позиции {pos}: {query[pos]!r}")
        pos = match.end()
        phrase, lparen, rparen, sym_op, word = match.groups()
        if phrase is not None:
            tokens.append(_QToken("PHRASE", phrase))
        elif lparen is not None:
            tokens.append(_QToken("LPAREN", "("))
        elif rparen is not None:
            tokens.append(_QToken("RPAREN", ")"))
        elif sym_op is not None:
            op = sym_op.strip()
            if op == "&":
                tokens.append(_QToken("AND", op))
            elif op in ("|", "||"):
                tokens.append(_QToken("OR", op))
            elif op == "~":
                tokens.append(_QToken("NOT", op))
        elif word is not None:
            upper = word.upper()
            if upper in ("AND", "OR", "NOT"):
                tokens.append(_QToken(upper, word))
            else:
                tokens.append(_QToken("WORD", word))
    return tokens


class _RecursiveDescentParser:
    """
    Грамматика:
      expr     -> or_expr
      or_expr  -> and_expr ( 'OR' and_expr )*
      and_expr -> not_expr ( ['AND'] not_expr )*
      not_expr -> 'NOT' not_expr | primary
      primary  -> '(' expr ')' | PHRASE | WORD
    """

    def __init__(self, tokens: list[_QToken]) -> None:
        self.tokens = tokens
        self.pos = 0

    def current(self) -> _QToken | None:
        return self.tokens[self.pos] if self.pos < len(self.tokens) else None

    def consume(self) -> _QToken:
        tok = self.tokens[self.pos]
        self.pos += 1
        return tok

    def parse(self) -> QueryNode:
        if not self.tokens:
            raise QuerySyntaxError("Пустой поисковый запрос.")
        node = self.parse_or()
        if self.current() is not None:
            raise QuerySyntaxError(f"Лишний токен в запросе: {self.current().value!r}")
        return node

    def parse_or(self) -> QueryNode:
        node = self.parse_and()
        while (tok := self.current()) is not None and tok.kind == "OR":
            self.consume()
            right = self.parse_and()
            node = Or(node, right)
        return node

    def parse_and(self) -> QueryNode:
        node = self.parse_not()
        while (tok := self.current()) is not None:
            if tok.kind == "AND":
                self.consume()
                right = self.parse_not()
                node = And(node, right)
            elif tok.kind in ("WORD", "PHRASE", "NOT", "LPAREN"):
                # Неявный AND: например, `python (async OR await) NOT java "event loop"`
                right = self.parse_not()
                node = And(node, right)
            else:
                break
        return node

    def parse_not(self) -> QueryNode:
        tok = self.current()
        if tok is not None and tok.kind == "NOT":
            self.consume()
            return Not(self.parse_not())
        return self.parse_primary()

    def parse_primary(self) -> QueryNode:
        tok = self.current()
        if tok is None:
            raise QuerySyntaxError("Неожиданный конец запроса.")

        if tok.kind == "LPAREN":
            self.consume()
            node = self.parse_or()
            closing = self.current()
            if closing is None or closing.kind != "RPAREN":
                raise QuerySyntaxError("Не закрыта скобка ')' в запросе.")
            self.consume()
            return node

        if tok.kind == "PHRASE":
            self.consume()
            norm_words = tuple(tokenize(tok.value))
            if not norm_words:
                raise QuerySyntaxError("Пустая фраза в кавычках.")
            if len(norm_words) == 1:
                return Term(norm_words[0])
            return Phrase(norm_words)

        if tok.kind == "WORD":
            self.consume()
            norm_words = list(tokenize(tok.value))
            if not norm_words:
                raise QuerySyntaxError(f"Некорректный термин: {tok.value!r}")
            node: QueryNode = Term(norm_words[0])
            for extra in norm_words[1:]:
                node = And(node, Term(extra))
            return node

        raise QuerySyntaxError(f"Ожидался термин или '(', получено: {tok.value!r}")


def parse_query(query: str) -> QueryNode:
    """Разбирает строку запроса в AST-дерево методом рекурсивного спуска."""
    tokens = _lex_query(query)
    return _RecursiveDescentParser(tokens).parse()