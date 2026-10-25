import ast
import json
from typing import Any

def extract_clean_text(content: Any) -> str:
    """Robustly extract clean Markdown/plain text from LangChain / Google GenAI message content.
    
    Handles:
    - Normal strings (with optional unicode / newline sanitization)
    - Google GenAI list of ContentBlocks: [{'type': 'text', 'text': '...'}]
    - Stringified Python representations of lists or dicts
    - Structured dictionary blocks with 'text' or 'content' keys
    """
    if not content:
        return ""

    if isinstance(content, list):
        parts = []
        for block in content:
            if isinstance(block, dict):
                if "text" in block:
                    parts.append(str(block["text"]))
                elif "content" in block:
                    parts.append(str(block["content"]))
                else:
                    parts.append(json.dumps(block, indent=2))
            elif isinstance(block, str):
                parts.append(block)
            elif hasattr(block, "text"):
                parts.append(str(getattr(block, "text")))
            else:
                parts.append(str(block))
        return "\n\n".join(p.strip() for p in parts if p.strip())

    if isinstance(content, dict):
        if "text" in content:
            return str(content["text"]).strip()
        if "content" in content:
            return str(content["content"]).strip()
        return json.dumps(content, indent=2)

    if isinstance(content, str):
        trimmed = content.strip()
        # Detect and parse stringified Python structures: e.g. "[{'type': 'text', 'text': ...}]"
        if (trimmed.startswith("[") and trimmed.endswith("]")) or (trimmed.startswith("{") and trimmed.endswith("}")):
            try:
                parsed = ast.literal_eval(trimmed)
                return extract_clean_text(parsed)
            except Exception:
                try:
                    parsed = json.loads(trimmed)
                    return extract_clean_text(parsed)
                except Exception:
                    pass
        # Unescape literal raw \n and \t sequences if they were serialized as escaped characters
        if "\\n" in trimmed and "\n" not in trimmed:
            try:
                trimmed = trimmed.encode("utf-8").decode("unicode_escape")
            except Exception:
                trimmed = trimmed.replace("\\n", "\n").replace("\\t", "\t")
        return trimmed

    return str(content)
