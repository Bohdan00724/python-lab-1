import html
import logging
import math
import re
import time
import uuid
from collections.abc import AsyncGenerator, Awaitable, Callable
from contextlib import asynccontextmanager
from dataclasses import replace
from pathlib import Path
from typing import Annotated

from fastapi import Depends, FastAPI, HTTPException, Query, Request, Response
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.templating import Jinja2Templates

from findex.index import Index
from findex.parser import QuerySyntaxError, parse_query
from findex.ranking import ScorerName, get_scorer
from findex.search import search
from findex.snippets import build_snippet
from findex.web.config import Settings, load_settings
from findex.web.schemas import (
    DocumentOut,
    HealthResponseOut,
    SearchHitOut,
    SearchResponseOut,
    StatsResponseOut,
    TermStatOut,
)

logger = logging.getLogger(__name__)

TEMPLATES_DIR = Path(__file__).resolve().parent / "templates"
templates = Jinja2Templates(directory=str(TEMPLATES_DIR))

_BOLD_SNIPPET_RE = re.compile(r"\*\*(.+?)\*\*")


def render_highlighted_snippet(snippet: str) -> str:
    """Безпечно екранує HTML та перетворює **термін** у <mark>термін</mark>."""
    escaped = html.escape(snippet)
    return _BOLD_SNIPPET_RE.sub(r"<mark>\1</mark>", escaped)


def get_settings(request: Request) -> Settings:
    settings: Settings | None = getattr(request.app.state, "settings", None)
    return settings if settings is not None else load_settings()


def get_optional_index(request: Request) -> Index | None:
    return getattr(request.app.state, "index", None)


def get_required_index(
    index: Annotated[Index | None, Depends(get_optional_index)],
) -> Index:
    if index is None:
        raise HTTPException(status_code=503, detail="Search index is not loaded")
    return index


def _read_document_content(index: Index, doc_id: int) -> str:
    raw_path = index.doc_path(doc_id)
    if raw_path and not raw_path.startswith(("http://", "https://")):
        file_path = Path(raw_path)
        if file_path.is_file():
            return file_path.read_text(encoding="utf-8")

    terms_in_doc: list[str] = []
    for term, plist in index.items():
        for posting in plist:
            if posting.doc_id == doc_id:
                terms_in_doc.append(f"{term} (tf={posting.frequency})")
                break
    return "Індексовані терміни документа: " + ", ".join(sorted(terms_in_doc))


def _execute_paginated_search(
    index: Index,
    q: str,
    k: int,
    scorer: ScorerName,
    page: int,
) -> SearchResponseOut:
    t_start = time.perf_counter()
    all_ranked = search(
        index=index,
        query=q,
        scorer=get_scorer(scorer),
        engine="merge",
        top_k=max(index.num_docs, page * k),
        with_snippets=False,
    )
    total = len(all_ranked)
    total_pages = max(1, math.ceil(total / k)) if total > 0 else 1
    offset = (page - 1) * k
    page_slice = all_ranked[offset : offset + k]

    ast = parse_query(q)
    score_terms = list(dict.fromkeys(ast.positive_terms()))
    phrases = list(ast.phrases())

    hits: list[SearchHitOut] = []
    for item in page_slice:
        snippet = build_snippet(item.path, score_terms, phrases=phrases, window=80)
        enriched = replace(item, snippet=snippet)
        hits.append(
            SearchHitOut(
                doc_id=enriched.doc_id,
                score=round(enriched.score, 6),
                path=enriched.path,
                snippet=enriched.snippet,
            )
        )

    elapsed_ms = (time.perf_counter() - t_start) * 1000.0
    return SearchResponseOut(
        query=q,
        scorer=scorer,
        page=page,
        k=k,
        total=total,
        total_pages=total_pages,
        has_prev=page > 1,
        has_next=offset + k < total,
        elapsed_ms=round(elapsed_ms, 3),
        results=hits,
    )


def create_app(settings: Settings | None = None) -> FastAPI:
    cfg = settings if settings is not None else load_settings()

    @asynccontextmanager
    async def lifespan(application: FastAPI) -> AsyncGenerator[None, None]:
        application.state.settings = cfg
        if getattr(application.state, "index", None) is None and cfg.index_path.is_file():
            logger.info("Завантаження індексу з %s", cfg.index_path)
            application.state.index = Index.load_json(cfg.index_path)
            logger.info("Індекс завантажено: %r", application.state.index)
        else:
            if getattr(application.state, "index", None) is None:
                logger.warning("Файл індексу не знайдено за шляхом %s", cfg.index_path)
                application.state.index = None
        yield
        loaded_idx: Index | None = getattr(application.state, "index", None)
        if loaded_idx is not None:
            loaded_idx.clear()
            application.state.index = None

    application = FastAPI(
        title=cfg.app_title,
        version="0.7.0",
        lifespan=lifespan,
    )
    application.state.settings = cfg
    application.state.index = None

    @application.middleware("http")
    async def request_context_middleware(
        request: Request,
        call_next: Callable[[Request], Awaitable[Response]],
    ) -> Response:
        request_id = request.headers.get("X-Request-ID") or uuid.uuid4().hex[:12]
        request.state.request_id = request_id
        try:
            response = await call_next(request)
            response.headers["X-Request-ID"] = request_id
            return response
        except Exception:
            logger.exception(
                "Неочікуваний виняток під час обробки запиту [request_id=%s] %s %s",
                request_id,
                request.method,
                request.url.path,
            )
            return JSONResponse(
                status_code=500,
                content={
                    "detail": "Internal Server Error",
                    "request_id": request_id,
                },
                headers={"X-Request-ID": request_id},
            )

    @application.get("/health", response_model=HealthResponseOut)
    def health_endpoint(
        response: Response,
        index: Annotated[Index | None, Depends(get_optional_index)],
    ) -> HealthResponseOut:
        if index is None:
            response.status_code = 503
            return HealthResponseOut(status="unavailable", index_loaded=False, num_docs=0)
        return HealthResponseOut(
            status="ok",
            index_loaded=True,
            num_docs=index.num_docs,
        )

    @application.get("/stats", response_model=StatsResponseOut)
    def stats_endpoint(
        index: Annotated[Index, Depends(get_required_index)],
    ) -> StatsResponseOut:
        top_keys = sorted(index.keys(), key=lambda t: index.df(t), reverse=True)[:10]
        top_terms = [TermStatOut(term=t, df=index.df(t)) for t in top_keys]
        return StatsResponseOut(
            num_docs=index.num_docs,
            num_terms=len(index),
            avg_doc_length=round(index.avg_doc_length, 3),
            store_positions=index.store_positions,
            top_terms=top_terms,
        )

    @application.get("/docs/{doc_id}", response_model=DocumentOut)
    def get_document_endpoint(
        doc_id: int,
        index: Annotated[Index, Depends(get_required_index)],
    ) -> DocumentOut:
        if doc_id not in index.all_doc_ids():
            raise HTTPException(status_code=404, detail=f"Document {doc_id} not found")
        content = _read_document_content(index, doc_id)
        return DocumentOut(
            doc_id=doc_id,
            path=index.doc_path(doc_id),
            length=index.doc_length(doc_id),
            content=content,
        )

    @application.get("/search", response_model=SearchResponseOut)
    def search_endpoint(
        index: Annotated[Index, Depends(get_required_index)],
        q: Annotated[str, Query(min_length=1, max_length=500, description="Пошуковий запит")],
        k: Annotated[int, Query(ge=1, le=50, description="Кількість результатів на сторінку")] = 10,
        scorer: Annotated[ScorerName, Query(description="Алгоритм: bm25 або tfidf")] = "bm25",
        page: Annotated[int, Query(ge=1, le=1000, description="Номер сторінки")] = 1,
    ) -> SearchResponseOut:
        try:
            return _execute_paginated_search(index=index, q=q, k=k, scorer=scorer, page=page)
        except (QuerySyntaxError, ValueError) as exc:
            raise HTTPException(status_code=422, detail=str(exc)) from None

    @application.get("/", response_class=HTMLResponse)
    def home_page(
        request: Request,
        index: Annotated[Index | None, Depends(get_optional_index)],
        q: Annotated[str, Query(max_length=500)] = "",
        k: Annotated[int, Query(ge=1, le=50)] = 10,
        scorer: Annotated[ScorerName, Query()] = "bm25",
        page: Annotated[int, Query(ge=1, le=1000)] = 1,
    ) -> HTMLResponse:
        current_cfg = get_settings(request)
        query_clean = q.strip()
        error_msg = ""
        search_data: SearchResponseOut | None = None

        if index is None:
            error_msg = "Індекс пошуку ще не завантажено (перевірте INDEX_PATH)."
        elif query_clean:
            try:
                search_data = _execute_paginated_search(
                    index=index, q=query_clean, k=k, scorer=scorer, page=page
                )
            except (QuerySyntaxError, ValueError) as exc:
                error_msg = f"Помилка синтаксису запиту: {exc}"

        rendered_results: list[dict[str, object]] = []
        if search_data is not None:
            for hit in search_data.results:
                rendered_results.append(
                    {
                        "doc_id": hit.doc_id,
                        "score": hit.score,
                        "path": hit.path,
                        "snippet_html": render_highlighted_snippet(hit.snippet),
                    }
                )

        context: dict[str, object] = {
            "request": request,
            "title": current_cfg.app_title,
            "num_docs": index.num_docs if index is not None else 0,
            "num_terms": len(index) if index is not None else 0,
            "q": query_clean,
            "k": k,
            "scorer": scorer,
            "page": page,
            "error_msg": error_msg,
            "total": search_data.total if search_data is not None else 0,
            "total_pages": search_data.total_pages if search_data is not None else 1,
            "has_prev": search_data.has_prev if search_data is not None else False,
            "has_next": search_data.has_next if search_data is not None else False,
            "elapsed_ms": search_data.elapsed_ms if search_data is not None else 0.0,
            "results": rendered_results,
        }
        return templates.TemplateResponse(request=request, name="index.html", context=context)

    return application


app = create_app()
