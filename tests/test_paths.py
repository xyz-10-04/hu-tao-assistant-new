"""路径与工作目录无关性测试。

这类 bug 的表现很隐蔽：从别的目录启动服务时，数据库和日志会被建到那个目录去，
用户看到的是"我昨天的笔记不见了"。所以用测试把它钉死。
"""
import os
import pathlib
import sys
import unittest

BASE_DIR = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BASE_DIR))


class TestPaths(unittest.TestCase):
    def test_database_path_is_absolute_and_in_project(self):
        import database

        db_path = pathlib.Path(database.DB_PATH)
        self.assertTrue(db_path.is_absolute(), "数据库路径必须是绝对路径")
        self.assertEqual(db_path.parent, BASE_DIR, "数据库必须落在项目根目录")

    def test_database_url_uses_forward_slashes(self):
        import database

        # SQLAlchemy 的 URL 用正斜杠更稳，反斜杠在部分平台会被当转义
        self.assertNotIn("\\", database.DATABASE_URL)
        self.assertTrue(database.DATABASE_URL.startswith("sqlite:///"))

    def test_rag_paths_are_absolute(self):
        import rag_build

        self.assertTrue(os.path.isabs(rag_build.KNOWLEDGE_DIR))
        self.assertTrue(os.path.isabs(rag_build.index_path(None)))
        self.assertEqual(pathlib.Path(rag_build.index_path("story")).parent, BASE_DIR)

    def test_rag_paths_survive_chdir(self):
        """切换工作目录后，路径不变——这是懒加载路径改动的核心目的。"""
        import rag_build

        before = rag_build.index_path(None)
        old = os.getcwd()
        try:
            os.chdir(pathlib.Path.home())
            self.assertEqual(rag_build.index_path(None), before)
        finally:
            os.chdir(old)


if __name__ == "__main__":
    unittest.main()
