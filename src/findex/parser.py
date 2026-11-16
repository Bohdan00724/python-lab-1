import re
from dataclasses import dataclass

from findex.core import tokenize
from findex.index import Index


class QuerySyntaxError(ValueError):
    """Помилка синтаксису пошукового запиту."""


class QueryNode:
    """Базовий вузол AST-дерева запиту з підтримкою операторів &, |, ~."""

    def __and__(self, other: "QueryNode") -> "And":
        return And(self, other)

    def __or__(self, other: "QueryNode") -> "Or":
        return Or(self, other)

    def __invert__(self) -> "Not":
        return Not(self)

    def evaluate(self, index: Index) -> set[int]:
        raise NotImplementedError

    def positive_terms(self) -> list[str]:
        """Повертає терміни (крім заперечень NOT) для підрахунку релевантності."""
        return []

    def phrases(self) -> list[tuple[str, ...]]:
        """Повертає точні фрази із запиту (для підсвічування сніпетів)."""
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
            raise ValueError("Фразовий пошук потребує індекс зі збереженими позиціями слів.")
        if len(self.terms) == 1:
            return Term(self.terms[0]).evaluate(index)

        postings_per_term: list[dict[int, set[int]]] = []
        for term in self.terms:
            if term not in index:
                return set()
            postings_per_term.append({p.doc_id: set(p.positions) for p in index[term]})

        doc_sets = [set(d.keys()) for d in postings_per_term]
        common_docs = doc_sets[0].intersection(*doc_sets[1:])
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
    kind: str
    value: str


_TOKEN_RE = re.compile(
    r"\s*(?:"
    r'"([^"]*)"'
    r"|(\()"
    r"|(\))"
    r"|(&|\|\||\||~)"
    r"|([\w'-]+)"
    r")"
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
            raise QuerySyntaxError(f"Неочікуваний символ у запиті на позиції {pos}: {query[pos]!r}")
        pos = match.end()
        phrase = match.group(1)
        lparen = match.group(2)
        rparen = match.group(3)
        sym_op = match.group(4)
        word = match.group(5)

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
            raise QuerySyntaxError("Порожній пошуковий запит.")
        node = self.parse_or()
        extra = self.current()
        if extra is not None:
            raise QuerySyntaxError(f"Зайвий токен у запиті: {extra.value!r}")
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
            raise QuerySyntaxError("Неочікуваний кінець запиту.")

        if tok.kind == "LPAREN":
            self.consume()
            node = self.parse_or()
            closing = self.current()
            if closing is None or closing.kind != "RPAREN":
                raise QuerySyntaxError("Не закрита дужка ')' у запиті.")
            self.consume()
            return node

        if tok.kind == "PHRASE":
            self.consume()
            norm_words = tuple(tokenize(tok.value))
            if not norm_words:
                raise QuerySyntaxError("Порожня фраза в лапках.")
            if len(norm_words) == 1:
                return Term(norm_words[0])
            return Phrase(norm_words)

        if tok.kind == "WORD":
            self.consume()
            norm_words = list(tokenize(tok.value))
            if not norm_words:
                raise QuerySyntaxError(f"Некоректний термін: {tok.value!r}")
            term_node: QueryNode = Term(norm_words[0])
            for extra_word in norm_words[1:]:
                term_node = And(term_node, Term(extra_word))
            return term_node

        raise QuerySyntaxError(f"Очікувався термін або '(', отримано: {tok.value!r}")


def parse_query(query: str) -> QueryNode:
    """Розбирає рядок запиту в AST-дерево методом рекурсивного спуску."""
    tokens = _lex_query(query)
    return _RecursiveDescentParser(tokens).parse()
