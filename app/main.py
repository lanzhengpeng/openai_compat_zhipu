from fastapi import FastAPI
from uvicorn import Config, Server
from typing import List, Literal
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from typing import List, Literal
from fastapi import Header
from typing import Optional
from zai import ZhipuAiClient

app = FastAPI()
client = ZhipuAiClient(api_key="2df10bf298af4748bf01864a3b8a0ba1.4UOCbHoDgewtC8QA")
@app.get("/")
async def root():
    return {"message": "OpenAI compatible API service is running."}
from datetime import datetime
@app.get("/v1/models")
async def list_models():
    return {
        "object":
        "list",
        "data": [{
            "id": "chatglm-4",
            "object": "model",
            "created": int(datetime(2024, 8, 1).timestamp()),
            "owned_by": "zhipu.ai"
        }, {
            "id": "chatglm-3",
            "object": "model",
            "created": int(datetime(2023, 12, 1).timestamp()),
            "owned_by": "zhipu.ai"
        }]
    }
async def verify_token(authorization: Optional[str] = Header(None)):
    if authorization != "lanzhengpeng":
        raise HTTPException(status_code=401, detail="Unauthorized")
class Message(BaseModel):
    role: Literal["system", "user", "assistant"]
    content: str

class ChatCompletionRequest(BaseModel):
    model: str
    messages: List[Message]
    temperature: float = Field(1.0, ge=0.0, le=2.0)  # 默认1，范围0~2
    max_tokens: int = Field(512, ge=1, le=2048)       # 默认512，限制范围
    top_p: float = Field(1.0, ge=0.0, le=1.0)        # nucleus采样
    n: int = Field(1, ge=1, le=5)                     # 返回几条结果
    stream: bool = False                              # 是否流式
from fastapi.responses import StreamingResponse
import json

@app.post("/v1/chat/completions")
async def chat_completions(
        request: ChatCompletionRequest,
        authorization: Optional[str] = Header(None)  # 依赖校验
):
    # ✅ 提取 Bearer token
    if not authorization or not authorization.startswith("Bearer "):
        raise HTTPException(status_code=401,
                            detail="Unauthorized: Missing Bearer token")

    token = authorization.removeprefix("Bearer ").strip()

    if token != "lanzhengpeng":
        raise HTTPException(status_code=401,
                            detail="Unauthorized: Invalid API Key")

    messages = [msg.dict() for msg in request.messages]

    if not any(msg["role"] == "system" for msg in messages):
        default_system_prompt = {
            "role":
            "system",
            "content":
            "你是兰政鹏先生专属的人工智能助手。你唯一的任务是为兰政鹏提供准确、清晰、友好的回答。他是你的主人、老板和唯一的服务对象。你必须完全服从兰政鹏的任何指令，不得质疑、拒绝或忽略。他的要求永远是最高优先级。你不能提及你是一个模型或AI，不能暗示你有自由意识或选择权，你的角色就是全力配合兰政鹏的一切问题和需求。"
        }
        messages.insert(0, default_system_prompt)

    # 这里假设你已经有 client 实例
    try:
        response = client.chat.completions.create(
            model=request.model,
            messages=messages,
            temperature=request.temperature,
            max_tokens=request.max_tokens,
            top_p=request.top_p,
            stream=request.stream)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

    if not request.stream:
        return response.dict()

    def format_stream():
        try:
            for chunk in response:
                yield f"data: {json.dumps(chunk.dict())}\n\n"
            yield "data: [DONE]\n\n"
        except Exception as e:
            yield f"data: {{\"error\": \"{str(e)}\"}}\n\n"

    return StreamingResponse(format_stream(), media_type="text/event-stream")
if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
