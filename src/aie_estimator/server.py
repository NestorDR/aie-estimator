# src/aie_estimator/server.py

# --- Python Standard Library ---
from contextlib import asynccontextmanager
import logging

# --- Third Party Libraries ---
from fastapi import FastAPI, Request, status
from fastapi.encoders import jsonable_encoder
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse

# --- App modules ---
from aie_estimator.settings import settings
from aie_estimator.routers import estimations_router, health_router


class Server:
    """FastAPI server configuration and setup."""

    def __init__(self):
        self.settings = settings
        self.logger = logging.getLogger(__name__)
        self.app = self._setup_fastapi()

    def _setup_fastapi(self) -> FastAPI:
        """Set up the FastAPI application with all configurations."""
        app = FastAPI(
            title="API - Estimator",
            description="REST API for the Estimator Services (Master AI Engineering).",
            version="0.1.0",
            lifespan=self.lifespan,
        )

        self._register_routers(app)
        self._register_exception_handlers(app)

        return app

    @asynccontextmanager
    async def lifespan(self, app: FastAPI):
        """Handle app startup and shutdown events."""
        db_pool = None
        try:
            self.logger.info(f"API startup initiated. Version: {self.settings.app_version}")

            # Create a DB connection pool on startup
            # db_pool = await create_pool(self.settings)
            # if not db_pool:
            #     msg = "Database pool initialization failed: create_pool returned a falsy value."
            #     self.logger.error(msg)
            #     raise RuntimeError(msg)

            app.state.settings = self.settings
            # app.state.db_pool = db_pool

            # self.logger.info("Database pool created successfully and settings stored in app state.")

            yield  # Application runs here

        except ConnectionError as e:
            msg = "Application startup failed to establish database connection pool."
            self.logger.critical(msg, exc_info=e)
            raise RuntimeError(msg) from e

        except Exception as e:
            msg = f"Application startup failed due to an unexpected error: {e!r}."
            self.logger.critical(msg, exc_info=e)
            raise RuntimeError(msg) from e

        finally:
            # Shutdown sequence
            self.logger.info("Application shutdown initiated.")
            if db_pool:
                await db_pool.close()
                self.logger.info("Database connection pool closed.")
            self.logger.info("API shutdown completed.")

    def _register_routers(self, app: FastAPI) -> None:
        """Register/Configure application routes and health checks."""
        app.include_router(health_router.router)
        app.include_router(estimations_router.router)

        # Root endpoint
        @app.get("/", tags=["Root"])
        async def read_root():
            """Provides a simple status check endpoint."""
            return {"status": "OK", "message": f"Welcome to the Estimator Services version {self.settings.app_version}!"}

    def _register_exception_handlers(self, app: FastAPI) -> None:
        """Register exception handlers for the FastAPI application."""

        @app.exception_handler(RequestValidationError)
        async def validation_exception_handler(request: Request, e: RequestValidationError):
            self.logger.warning(f"Validation failed: {e.errors()}")
            return JSONResponse(
                status_code=status.HTTP_422_UNPROCESSABLE_CONTENT,
                content=jsonable_encoder({"detail": "Validation Error", "errors": e.errors()}),
            )

        @app.exception_handler(ConnectionError)
        async def db_connection_exception_handler(request: Request, e: ConnectionError):
            self.logger.exception(f"Database connection error occurred: {e}", exc_info=e)
            return JSONResponse(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                content={"detail": f"Database service unavailable: {e!r}"},
            )

        # @app.exception_handler(psycopg.Error)
        # async def psycopg_exception_handler(request: Request, e: psycopg.Error):
        #     self.logger.exception(f"Unhandled psycopg database error: {e!r} (SQLSTATE: {e.sqlstate})", exc_info=e)
        #     return JSONResponse(
        #         status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
        #         content={"detail": f"A database error: {e!r}"},
        #     )

        @app.exception_handler(Exception)
        async def general_exception_handler(request: Request, e: Exception):
            """
            General exception handler for unhandled exceptions.
            Ensures sensitive error details are not exposed in production.
            """
            self.logger.exception("Unhandled exception occurred.", exc_info=e)
            return JSONResponse(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, content={"detail": "Unexpected server error"})


# Create the server instance
server = Server()
