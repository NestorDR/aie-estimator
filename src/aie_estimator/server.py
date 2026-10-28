# src/aie_estimator/server.py

# --- Python Standard Library ---
from contextlib import asynccontextmanager
from typing import Final

# --- Third Party Libraries ---
from fastapi import FastAPI, Request, status
from fastapi.encoders import jsonable_encoder
from fastapi.exceptions import RequestValidationError
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
import structlog

# --- App modules ---
from aie_estimator.routers import estimations_router, health_router
from aie_estimator.services.llm_service import LLMServiceError
from aie_estimator.settings import Settings, get_settings


class Server:
    """FastAPI server configuration and setup."""

    def __init__(self):
        """Initialize server settings, structured logger, and the FastAPI application."""
        self.settings: Settings = get_settings()
        self.log: Final[structlog.stdlib.BoundLogger] = structlog.get_logger(__name__)
        self.app = self._setup_fastapi()

    def _setup_fastapi(self) -> FastAPI:
        """
        Set up the FastAPI application with all configurations.

        :return: Configured FastAPI application instance.
        """
        app = FastAPI(
            title="API - Estimator",
            description="REST API for the Estimator Services (Master AI Engineering).",
            version="0.1.0",
            lifespan=self.lifespan,
        )

        self._register_routers(app)
        self._register_exception_handlers(app)
        self._register_cors_middleware(app)

        return app

    @asynccontextmanager
    async def lifespan(self, app: FastAPI):
        """
        Handle app startup and shutdown events.

        :param app: FastAPI application instance.
        :return: Asynchronous context manager managing application lifespan.
        """
        db_pool = None
        try:
            self.log.info("api_startup_initiated", app_version=self.settings.app_version)

            # Create a DB connection pool on startup
            # db_pool = await create_pool(self.settings)
            # if not db_pool:
            #     self.log.critical("database_pool_initialization_failed", error=f"create_pool returned a falsy value. {e!r}", exc_info=e)
            #     raise RuntimeError("Database pool initialization failed: create_pool returned a falsy value.")

            app.state.settings = self.settings
            # app.state.db_pool = db_pool

            # self.log.info("database_pool_and_app_state_created_successfully.")

            yield  # Application runs here

        except ConnectionError as e:
            self.log.critical("database_connection_pool_failed", error=repr(e), exc_info=e)
            raise RuntimeError("Application startup failed to establish database connection pool.") from e

        except Exception as e:
            self.log.critical("application_startup_failed", error=repr(e), exc_info=e)
            raise RuntimeError(f"Application startup failed due to an unexpected error: {e!r}.") from e

        finally:
            # Shutdown sequence
            self.log.info("application_shutdown_initiated")
            if db_pool:
                await db_pool.close()
                self.log.info("database_connection_pool_closed")
            self.log.info("api_shutdown_completed")

    def _register_routers(self, app: FastAPI) -> None:
        """
        Register and configure application routes and health checks.

        :param app: FastAPI application instance to register routers on.
        """
        app.include_router(health_router.router)
        app.include_router(estimations_router.router)

        # Root endpoint
        @app.get("/", tags=["Root"])
        async def read_root():
            """
            Provides a simple status check endpoint.

            :return: Dictionary containing service status and welcome metadata.
            """
            return {
                "status": "OK",
                "message": f"Welcome to the Estimator Services version {self.settings.app_version}!",
            }

    def _register_exception_handlers(self, app: FastAPI) -> None:
        """
        Register exception handlers for the FastAPI application.

        :param app: FastAPI application instance to register exception handlers on.
        """

        @app.exception_handler(RequestValidationError)
        async def validation_exception_handler(request: Request, e: RequestValidationError):
            """
            Handle request validation errors and return structured HTTP 422 responses.

            :param request: Incoming HTTP request that caused the validation error.
            :param e: Raised request validation error instance.

            :return: JSONResponse containing serialized validation error details.
            """
            self.log.warning("request_validation_failed", errors=e.errors())
            return JSONResponse(
                status_code=status.HTTP_422_UNPROCESSABLE_CONTENT,
                content=jsonable_encoder({"detail": "Validation Error", "errors": e.errors()}),
            )

        @app.exception_handler(ConnectionError)
        async def db_connection_exception_handler(request: Request, e: ConnectionError):
            """
            Handle database connection failures and return structured HTTP 503 responses.

            :param request: Incoming HTTP request during which the error occurred.
            :param e: Database connection error instance.

            :return: JSONResponse indicating database service unavailability.
            """
            self.log.exception("database_connection_error", error=repr(e), exc_info=e)
            return JSONResponse(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                content={"detail": f"Database service unavailable: {e!r}"},
            )

        # @app.exception_handler(psycopg.Error)
        # async def psycopg_exception_handler(request: Request, e: psycopg.Error):
        #     self.log.exception("unhandled_psycopg_database_error", error=f"{e!r} (SQLSTATE: {e.sqlstate})", exc_info=e)
        #     return JSONResponse(
        #         status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
        #         content={"detail": f"A database error: {e!r}"},
        #     )

        @app.exception_handler(LLMServiceError)
        async def llm_service_exception_handler(request: Request, e: LLMServiceError):
            """
            Handle upstream LLM service errors and return sanitized HTTP 500 responses.

            :param request: Incoming HTTP request triggering the service failure.
            :param e: Upstream LLM domain service exception.

            :return: JSONResponse containing a generic user-facing error message.
            """
            self.log.exception("llm_service_error_encountered", error=repr(e), exc_info=e)
            return JSONResponse(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                content={"detail": "An error occurred while generating the estimation. Please try again later."},
            )

        @app.exception_handler(Exception)
        async def general_exception_handler(request: Request, e: Exception):
            """
            General exception handler for unhandled exceptions.
            Ensures sensitive error details are not exposed in production.

            :param request: Incoming HTTP request.
            :param e: Unhandled exception instance.

            :return: JSONResponse with HTTP 500 status code.
            """
            self.log.exception("unhandled_server_exception", error=repr(e), exc_info=e)
            return JSONResponse(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                content={"detail": "Unexpected server error"},
            )

    def _register_cors_middleware(self, app: FastAPI) -> None:
        """
        Register CORS middleware with specific origins, methods, and headers.

        :param app: FastAPI application instance to attach middleware to.
        """
        origins = [
            "http://localhost",
            "http://127.0.0.1",
        ]

        app.add_middleware(
            CORSMiddleware,
            allow_origins=origins,
            allow_credentials=True,
            allow_methods=["*"],  # Allow all methods
            allow_headers=["*"],  # Allow all headers
        )

        self.log.info("cors_middleware_registered")


# Create the server instance
server = Server()
