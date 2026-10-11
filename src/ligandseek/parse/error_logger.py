"""Write structured parser error logs under the supplied directory."""

import os
import json
from pathlib import Path
from datetime import datetime
from typing import Optional


class ProcessingError(RuntimeError):


    def __init__(self, error_type: str, error_detail: str, logged: bool = False):
        super().__init__(f"{error_type}: {error_detail}")
        self.error_type = error_type
        self.error_detail = error_detail
        self.logged = logged


def return_error_info(
    file_path: str,
    line: int,
    error_type: str,
    error_detail: str,
    output_dir: str,
    sample_id: Optional[str] = None
) -> None:


    if sample_id is None:
        try:
            
            sample_id = Path(file_path).stem
        except Exception:
            sample_id = "unknown"


    error_log_dir = Path(output_dir) / "error_logs"
    try:
        error_log_dir.mkdir(parents=True, exist_ok=True)
    except Exception as e:
        print(f"[Error] Failed to create error log directory: {e}")
        return
    
    error_log_path = error_log_dir / f"error_{sample_id}.json"
    
    error_entry = {
        "sample_id": sample_id,
        "file": str(file_path),
        "line": line,
        "error_type": error_type,
        "error_detail": error_detail,
        "timestamp": datetime.now().isoformat()
    }


    try:
        
        existing_logs = []
        if error_log_path.exists():
            try:
                with open(error_log_path, 'r', encoding='utf-8') as f:
                    content = f.read().strip()
                    if content:
                        existing_logs = json.loads(content)
            except (json.JSONDecodeError, IOError):
                
                existing_logs = []


        existing_logs.append(error_entry)


        with open(error_log_path, 'w', encoding='utf-8') as f:
            json.dump(existing_logs, f, indent=2, ensure_ascii=False)
            
    except Exception as e:
        
        print(f"[Error] Failed to write error log for {sample_id}: {e}")
        print(f"  Error type: {error_type}")
        print(f"  Error detail: {error_detail}")


def return_error_and_raise(
    file_path: str,
    line: int,
    error_type: str,
    error_detail: str,
    output_dir: str,
    sample_id: Optional[str] = None,
) -> None:


    return_error_info(
        file_path=file_path,
        line=line,
        error_type=error_type,
        error_detail=error_detail,
        output_dir=output_dir,
        sample_id=sample_id,
    )
    raise ProcessingError(error_type=error_type, error_detail=error_detail, logged=True)


class ErrorType:

    MISSING_BACKBONE = "MISSING_BACKBONE"       
    NO_LIGAND = "NO_LIGAND"                     
    NO_BINDING_SITE = "NO_BINDING_SITE"         
    PARSE_ERROR = "PARSE_ERROR"                 
    INVALID_RESIDUE = "INVALID_RESIDUE"         
    INVALID_ELEMENT = "INVALID_ELEMENT"         
    FILE_NOT_FOUND = "FILE_NOT_FOUND"           
    EMPTY_STRUCTURE = "EMPTY_STRUCTURE"         
    DUPLICATE_ATOM = "DUPLICATE_ATOM"           
    INCOMPLETE_RESIDUE = "INCOMPLETE_RESIDUE"   
    INCOMPLETE_BACKBONE = "INCOMPLETE_BACKBONE" 
    INVALID_CONFIGURATION = "INVALID_CONFIGURATION" 
    INVALID_LOCAL_FRAME = "INVALID_LOCAL_FRAME"
