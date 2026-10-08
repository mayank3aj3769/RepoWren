"""HTTP routes for health, readiness, and streaming chat."""

from fastapi import APIRouter, HTTPException, Request, status
from fastapi.responses import StreamingResponse

from local_agent.api.schemas import ChatRequest, StatusResponse
from local_agent.inference.llama_cpp import LlamaCppClient
from local_agent.services.chat import ChatService


router = APIRouter()


def _inference(request: Request) -> LlamaCppClient:
    return request.app.state.inference


@router.get("/status", response_model=StatusResponse)
async def status_endpoint(request: Request) -> StatusResponse:
    ready = await _inference(request).is_ready()
    return StatusResponse(inference="ready" if ready else "unavailable")


@router.post("/v1/chat/stream")
async def stream_chat(request: Request, body: ChatRequest) -> StreamingResponse:
    inference = _inference(request)
    if not await inference.is_ready():
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="The local llama.cpp server is not ready.",
        )

    service = ChatService(inference)
    return StreamingResponse(
        service.stream_events(body),
        media_type="application/x-ndjson",
        headers={"X-Content-Type-Options": "nosniff"},
    )
