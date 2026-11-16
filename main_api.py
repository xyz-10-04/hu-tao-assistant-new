import json
import logging
import time
import uuid
from logging.handlers import RotatingFileHandler

from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from langchain_core.messages import AIMessage, HumanMessage
from pydantic import BaseModel, Field

from main import agent
from memory import load_memory, save_memory
from tools_api import calculate, get_current_time, read_notes, roll_dice, save_note

# 创建 logger
logger = logging.getLogger("huTao")   # "huTao" 可以改成你喜欢的名字，但不是必须改
logger.setLevel(logging.INFO)   # 这一行保留，它控制了日志级别

# 文件处理器（轮转）
file_handler = RotatingFileHandler("huTao.log", maxBytes=10*1024*1024, backupCount=5)
file_handler.setLevel(logging.INFO)

# 控制台处理器
console_handler = logging.StreamHandler()
console_handler.setLevel(logging.INFO)

# 格式
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
file_handler.setFormatter(formatter)
console_handler.setFormatter(formatter)

# 添加处理器
logger.addHandler(file_handler)
logger.addHandler(console_handler)

# 然后你就可以在代码中使用 logger.info(...)、logger.error(...) 等

app = FastAPI(
    title="胡桃助手API",
    description="往生堂第七十七代堂主胡桃为你服务——掷骰子、查时间、算算术、记笔记，样样精通。",
    version="1.0.0"
)
@app.get("/roll")
async def api_roll():
    return {"result": roll_dice()}

@app.get("/time")
async def api_time():
    return {"time": get_current_time()}

@app.get("/health")
async def api_health():
    """给部署/监控用的健康检查：不触发模型加载，能秒回。"""
    return {"status": "ok", "service": "hutao-assistant"}


def _calc_result(expr: str):
    result = calculate(expr)
    # calculate 把异常吞成了字符串，这里翻译成规范的 4xx，前端才能区分“算错”和“服务挂了”
    if isinstance(result, str) and result.startswith("错误"):
        raise HTTPException(status_code=400, detail=result.split("：", 1)[-1])
    return {"result": result}


class CalcRequest(BaseModel):
    expr: str = Field(..., min_length=1, max_length=100, description="算式，如 (3+8)*2")


@app.get("/calc")
async def api_calc(expr: str = Query(..., min_length=1, max_length=100)):
    return _calc_result(expr)


@app.post("/calc")
async def api_calc_post(request: CalcRequest):
    """推荐用 POST：算式放在 body 里，不会被 URL 编码把 '+' 变成空格。"""
    return _calc_result(request.expr)

@app.post("/note")
async def api_save_note(content: str):
    result = save_note(content)
    return {"status": "saved" if result else "failed"}

@app.get("/notes")
async def api_read_notes():
    notes = read_notes()
    return {"notes": notes}

class MemoryUpdate(BaseModel):
    data: dict  # 前端传入的记忆数据


@app.get("/memory")
async def api_get_memory():
    """获取当前记忆"""
    memory = load_memory()
    return {"memory": memory}

@app.post("/memory")
async def api_update_memory(update: MemoryUpdate):
    """更新记忆（覆盖写入）"""
    try:
        save_memory(update.data)
        return {"status": "saved", "memory": update.data}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


class ChatRequest(BaseModel):
    message: str = Field(..., min_length=1, max_length=500, description="用户输入的消息")
    history: list = Field(default_factory=list, description="历史对话记录，格式为 [{'role': 'user', 'content': '...'}, {'role': 'assistant', 'content': '...'}]")


def _text_of(content):
    """从 chunk.content 里取出纯文本。

    新版 langchain 的 content 可能是 str，也可能是 [{'type': 'text', 'text': ...}] 这样的块列表，
    直接 f-string 会把列表的 repr 打进前端，这里统一成字符串。
    """
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts = []
        for block in content:
            if isinstance(block, str):
                parts.append(block)
            elif isinstance(block, dict) and block.get("type") == "text":
                parts.append(block.get("text", ""))
        return "".join(parts)
    return "" if content is None else str(content)


import asyncio

@app.post("/chat")
async def chat_endpoint(request: ChatRequest):
    # 🚨 确保这行在函数最开头，不要漏了
    request_id = str(uuid.uuid4())[:8]
    start_time = time.time()
    logger.info(f"[{request_id}] 收到请求: {request.message[:50]}...")

    try:
        # 1. 解析历史记录
        messages = []
        for msg in request.history:
            if msg.get("role") == "user":
                messages.append(HumanMessage(content=msg.get("content")))
            elif msg.get("role") == "assistant":
                messages.append(AIMessage(content=msg.get("content")))
        
        # 2. 拼入最新一句话
        messages.append(HumanMessage(content=request.message))
        
        # 3. 截断
        if len(messages) > 20:
            messages = messages[-20:]

        # 4. 调 Agent
        result = await asyncio.wait_for(
            asyncio.to_thread(
                agent.invoke,
                {"messages": messages}
            ),
            timeout=90.0
        )
        
        reply = result["messages"][-1].content
        elapsed = time.time() - start_time
        logger.info(f"[{request_id}] 回复成功，耗时: {elapsed:.2f}s")
        return {"reply": reply}
    
    except asyncio.TimeoutError:
        logger.warning(f"[{request_id}] 请求超时: {request.message[:30]}...")
        raise HTTPException(status_code=504, detail="请求超时，请稍后再试")
    except Exception as e:
        logger.error(f"[{request_id}] 处理请求失败: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/chat/stream")
async def chat_stream_endpoint(request: ChatRequest):
    # 🚨 确保这行在函数最开头，不要漏了
    request_id = str(uuid.uuid4())[:8]
    logger.info(f"[{request_id}] 收到流式请求: {request.message[:50]}...")

    if len(request.message) > 500:
        async def error_stream():
            yield f"data: {json.dumps('输入过长，请缩短后再试', ensure_ascii=False)}\n\n"
            yield "data: [DONE]\n\n"
        return StreamingResponse(error_stream(), media_type="text/event-stream")

    async def generate():
        try:
            # 1. 解析历史记录
            messages = []
            for msg in request.history:
                if msg.get("role") == "user":
                    messages.append(HumanMessage(content=msg.get("content")))
                elif msg.get("role") == "assistant":
                    messages.append(AIMessage(content=msg.get("content")))
            
            # 2. 拼入最新一句话
            messages.append(HumanMessage(content=request.message))
            
            # 3. 截断
            if len(messages) > 20:
                messages = messages[-20:]

            # 4. 流式调 Agent
            async for event in agent.astream_events(
                {"messages": messages},
                version="v2"
            ):
                if event["event"] == "on_chat_model_stream":
                    chunk = event["data"]["chunk"]
                    text = _text_of(chunk.content)
                    if text:
                        # 用 JSON 包装增量：正文里的换行交给 json 转义，
                        # 不会因为 SSE 按行分帧而丢掉（旧版会把多段回答挤成一段）
                        yield f"data: {json.dumps(text, ensure_ascii=False)}\n\n"
            yield "data: [DONE]\n\n"
        except Exception as e:
            logger.error(f"[{request_id}] 流式处理失败: {e}")
            yield f"data: {json.dumps('处理失败: ' + str(e), ensure_ascii=False)}\n\n"
            yield "data: [DONE]\n\n"
            
    return StreamingResponse(generate(), media_type="text/event-stream")


app.add_middleware(
    CORSMiddleware,
    # 注意：allow_origins=["*"] 与 allow_credentials=True 是浏览器会直接拒绝的非法组合，
    # 需要携带 cookie 时请把 origins 收窄成具体域名。
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)

