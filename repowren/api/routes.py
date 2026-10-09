"""HTTP routes for health, model status, and streaming chat."""

from fastapi import APIRouter, Request
from fastapi.responses import StreamingResponse

from repowren.api.schemas import ChatRequest, StatusResponse
from repowren.inference.base import InferenceBackend
from repowren.services.chat import ChatService


router = APIRouter()


def _inference(request: Request) -> InferenceBackend:
    return request.app.state.inference


@router.get("/status", response_model=StatusResponse)
async def status_endpoint(request: Request) -> StatusResponse:
    inference = _inference(request)
    await inference.check_ready()
    return StatusResponse(inference=inference.status, model=inference.model_id)


@router.post("/v1/chat/stream")
async def stream_chat(request: Request, body: ChatRequest) -> StreamingResponse:
    inference = _inference(request)
    service = ChatService(inference)
    return StreamingResponse(
        service.stream_events(body),
        media_type="application/x-ndjson",
        headers={"X-Content-Type-Options": "nosniff"},
    )
