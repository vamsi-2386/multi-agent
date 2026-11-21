"""
TripMate AI - Multi-Agent Travel Planning Assistant
Main FastAPI Application Entrypoint
"""

import time
import uuid
from contextlib import asynccontextmanager
from typing import AsyncGenerator

from fastapi import FastAPI, Request, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from app.api.router import api_router
from app.config import settings
from app.utils.logger import get_logger, request_id_ctx, setup_logging

logger = get_logger("app.main")


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncGenerator[None, None]:
    """Application lifespan manager: sets up resources at startup and tears down on shutdown."""
    setup_logging()
    logger.info(
        f"Starting {settings.APP_NAME} v{settings.APP_VERSION} "
        f"[env={settings.APP_ENV}, llm_provider={settings.LLM_PROVIDER}]"
    )
    try:
        from app.database.session import init_db
        await init_db()
    except Exception as db_err:
        logger.warning("Could not auto-initialize database tables: %s", db_err)
    yield
    logger.info(f"Shutting down {settings.APP_NAME}")


app = FastAPI(
    title=settings.APP_NAME,
    version=settings.APP_VERSION,
    description=(
        "Production-ready backend API for TripMate AI – an intelligent multi-agent "
        "travel planning assistant powered by LangGraph, FastAPI, and Pydantic."
    ),
    docs_url="/docs",
    redoc_url="/redoc",
    openapi_url="/openapi.json",
    lifespan=lifespan,
)

# CORS Middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.CORS_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.middleware("http")
async def request_context_middleware(request: Request, call_next):
    """
    Middleware to assign and track a unique request ID across async tasks
    and measure total request latency.
    """
    req_id = request.headers.get("X-Request-ID", str(uuid.uuid4()))
    token = request_id_ctx.set(req_id)
    start_time = time.perf_counter()

    try:
        response = await call_next(request)
        process_time = time.perf_counter() - start_time
        response.headers["X-Request-ID"] = req_id
        response.headers["X-Process-Time"] = f"{process_time:.4f}s"
        return response
    finally:
        request_id_ctx.reset(token)


@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception) -> JSONResponse:
    """Safe global exception handler: logs internal details, returns clean generic message."""
    logger.error(
        f"Unhandled exception during {request.method} {request.url.path}: {str(exc)}",
        exc_info=True,
    )
    return JSONResponse(
        status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
        content={
            "error": "InternalServerError",
            "message": "An unexpected error occurred while processing your request.",
            "request_id": request_id_ctx.get(),
        },
    )


# Root landing endpoint
@app.get(
    "/",
    tags=["Root"],
    summary="API Root Information",
    status_code=status.HTTP_200_OK,
)
async def root():
    return {
        "service": settings.APP_NAME,
        "version": settings.APP_VERSION,
        "environment": settings.APP_ENV,
        "documentation": "/docs",
        "health": f"{settings.API_V1_PREFIX}/health",
    }


# Mount versioned API routes
app.include_router(api_router, prefix=settings.API_V1_PREFIX)

