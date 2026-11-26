"""Rejects request bodies above a size limit while they arrive, not after they are read."""

from starlette.responses import PlainTextResponse
from starlette.types import ASGIApp, Message, Receive, Scope, Send


class _BodyTooLarge(Exception):
    pass


class BodySizeLimit:
    def __init__(self, app: ASGIApp, max_bytes: int) -> None:
        self.app = app
        self.max_bytes = max_bytes

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        if scope["type"] != "http":
            await self.app(scope, receive, send)
            return

        too_large = PlainTextResponse("request body is too large", status_code=413)
        content_length = dict(scope["headers"]).get(b"content-length", b"")
        if content_length.isdigit() and int(content_length) > self.max_bytes:
            await too_large(scope, receive, send)
            return

        received = 0
        exceeded = False
        response_started = False

        async def limited_receive() -> Message:
            nonlocal received, exceeded
            message = await receive()
            if message["type"] == "http.request":
                received += len(message.get("body", b""))
                if received > self.max_bytes:
                    exceeded = True
                    raise _BodyTooLarge
            return message

        async def guarded_send(message: Message) -> None:
            nonlocal response_started
            # FastAPI turns the error raised while reading into its own error response.
            # Once the limit was exceeded, that response is replaced by a 413.
            if exceeded:
                if message["type"] == "http.response.start" and not response_started:
                    response_started = True
                    await too_large(scope, receive, send)
                return
            response_started = response_started or message["type"] == "http.response.start"
            await send(message)

        try:
            await self.app(scope, limited_receive, guarded_send)
        except _BodyTooLarge:
            if not response_started:
                await too_large(scope, receive, send)
