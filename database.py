import os
from pathlib import Path
from typing import Generator

from sqlalchemy import create_engine
from sqlalchemy.orm import Session, sessionmaker

# 路径基于本文件位置，而不是当前工作目录——
# 否则从别的目录启动 uvicorn，数据库会被建到那个目录去，表现为"数据莫名其妙没了"
BASE_DIR = Path(__file__).resolve().parent
DB_PATH = BASE_DIR / "hutao.db"

# 用 as_posix() 把 Windows 反斜杠转成正斜杠，SQLAlchemy 的 URL 更认这个
DATABASE_URL = f"sqlite:///{DB_PATH.as_posix()}"

# echo=True 会把每条 SQL 都打到日志里，调试时用 SQL_ECHO=1 打开
engine = create_engine(DATABASE_URL, echo=os.getenv("SQL_ECHO") == "1")

# 创建会话工厂
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

# 依赖注入用：每次请求获取一个会话
def get_db() -> Generator[Session, None, None]:
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()