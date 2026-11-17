"""Chat history management using Unity Catalog Delta tables."""

import json
import logging
from datetime import datetime
from typing import List, Optional
from uuid import uuid4

logger = logging.getLogger(__name__)

CHAT_HISTORY_TABLE = "workspace.legal_assistant_lakebase.chat_history"


class ChatHistoryManager:
    """Manages chat history storage and retrieval using Unity Catalog."""

    def __init__(self):
        try:
            # In Databricks Apps, use SQL warehouse via SDK instead of Spark
            from databricks.sdk import WorkspaceClient
            self.w = WorkspaceClient()
            
            # Get SQL warehouse ID from app resources
            self.warehouse_id = "c99228c48630a173"  # From app resources
            
            logger.info(f"Initialized ChatHistoryManager with table: {CHAT_HISTORY_TABLE}")
        except Exception as e:
            logger.warning(f"Could not initialize ChatHistoryManager: {e}")
            self.w = None
            self.warehouse_id = None

    def save_message(self, session_id: str, role: str, content: str, metadata: Optional[dict] = None) -> bool:
        if not self.w or not self.warehouse_id:
            logger.warning("ChatHistoryManager not available, skipping message save")
            return False
        try:
            message_id = str(uuid4())
            timestamp = datetime.utcnow().isoformat()
            metadata_json = json.dumps(metadata) if metadata else "null"
            
            # Escape single quotes in content
            content_escaped = content.replace("'", "''")
            
            # Use SQL warehouse to insert data
            sql = f"""
                INSERT INTO {CHAT_HISTORY_TABLE} 
                (session_id, message_id, role, content, timestamp, metadata)
                VALUES (
                    '{session_id}',
                    '{message_id}',
                    '{role}',
                    '{content_escaped}',
                    '{timestamp}',
                    '{metadata_json}'
                )
            """
            
            # Execute via SQL warehouse
            self.w.statement_execution.execute_statement(
                warehouse_id=self.warehouse_id,
                statement=sql,
                wait_timeout="30s"
            )
            
            logger.info(f"Saved message {message_id} for session {session_id}")
            return True
        except Exception as e:
            logger.error(f"Failed to save message: {e}")
            return False

    def get_session_history(self, session_id: str, max_messages: int = 50) -> List[dict]:
        if not self.w or not self.warehouse_id:
            logger.warning("ChatHistoryManager not available, returning empty history")
            return []
        try:
            sql = f"""
                SELECT role, content, timestamp, metadata
                FROM {CHAT_HISTORY_TABLE}
                WHERE session_id = '{session_id}'
                ORDER BY timestamp ASC
                LIMIT {max_messages}
            """
            
            # Execute query via SQL warehouse
            result = self.w.statement_execution.execute_statement(
                warehouse_id=self.warehouse_id,
                statement=sql,
                wait_timeout="30s"
            )
            
            messages = []
            if result.result and result.result.data_array:
                for row in result.result.data_array:
                    message = {
                        "role": row[0],
                        "content": row[1],
                        "timestamp": row[2]
                    }
                    if row[3]:
                        try:
                            message["metadata"] = json.loads(row[3])
                        except json.JSONDecodeError:
                            pass
                    messages.append(message)
            
            logger.info(f"Retrieved {len(messages)} messages for session {session_id}")
            return messages
        except Exception as e:
            logger.error(f"Failed to retrieve session history: {e}")
            return []

    def format_for_agent(self, messages: List[dict]) -> List[dict]:
        return [{"role": msg["role"], "content": msg["content"]} for msg in messages]

    def save_conversation(self, session_id: str, user_message: str, assistant_response: str, user_metadata: Optional[dict] = None, assistant_metadata: Optional[dict] = None) -> bool:
        user_saved = self.save_message(session_id=session_id, role="user", content=user_message, metadata=user_metadata)
        assistant_saved = self.save_message(session_id=session_id, role="assistant", content=assistant_response, metadata=assistant_metadata)
        return user_saved and assistant_saved
