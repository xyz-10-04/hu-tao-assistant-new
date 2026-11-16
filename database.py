import os
from typing import Generator

from sqlalchemy import create_engine
from sqlalchemy.orm import Session, sessionmaker

# 创建数据库引擎（使用 SQLite）
DATABASE_URL = "sqlite:///./hutao.db"  # 会在项目根目录生成 hutao.db 文件
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