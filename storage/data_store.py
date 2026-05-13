import json
import uuid
import threading
import os
from datetime import datetime
from pathlib import Path
from typing import Optional


DATA_DIR   = Path(__file__).parent / "data"
DATA_DIR.mkdir(parents=True, exist_ok=True)

TICKETS_FILE  = DATA_DIR / "tickets.json"
ACCOUNTS_FILE = DATA_DIR / "accounts.json"
LOG_FILE      = DATA_DIR / "query_log.json"

_lock = threading.Lock()


def _read(path: Path) -> dict:
    if not path.exists():
        return {}
    with open(path) as f:
        return json.load(f)


def _write(path: Path, data: dict) -> None:
    with open(path, "w") as f:
        json.dump(data, f, indent=2, default=str)


def _now() -> str:
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")


def _ticket_id(prefix: str = "TKT") -> str:
    return f"{prefix}-{uuid.uuid4().hex[:6].upper()}"


PRIORITY_PREFIX = {"P1": "EMG", "P2": "DC", "P3": "SVC", "P4": "FB"}
PRIORITY_SLA    = {"P1": 1, "P2": 4, "P3": 24, "P4": 48}

STATUS_OPEN        = "Open"
STATUS_IN_PROGRESS = "In Progress"
STATUS_RESOLVED    = "Resolved"
STATUS_CLOSED      = "Closed"


def create_ticket(
    session_id:          str,
    query:               str,
    complaint_type:      str,
    priority:            str,          # P1–P4
    monetary_impact:     str,
    escalation_required: bool,
    sla_hours:           int,
    reason:              str,
    amount:              Optional[str] = None,
    date:                Optional[str] = None,
) -> dict:
    
    prefix    = PRIORITY_PREFIX.get(priority, "TKT")
    ticket_id = _ticket_id(prefix)

    ticket = {
        "ticket_id":           ticket_id,
        "session_id":          session_id,
        "query":               query,
        "complaint_type":      complaint_type,
        "priority":            priority,
        "monetary_impact":     monetary_impact,
        "escalation_required": escalation_required,
        "sla_hours":           sla_hours,
        "reason":              reason,
        "amount":              amount,
        "transaction_date":    date,
        "status":              STATUS_OPEN,
        "created_at":          _now(),
        "updated_at":          _now(),
        "resolution_note":     None,
    }

    with _lock:
        store = _read(TICKETS_FILE)
        store[ticket_id] = ticket
        _write(TICKETS_FILE, store)

    return ticket


def get_ticket(ticket_id: str) -> Optional[dict]:
    """Fetch a ticket by ID. Returns None if not found."""
    with _lock:
        store = _read(TICKETS_FILE)
    return store.get(ticket_id.upper())


def update_ticket_status(ticket_id: str, status: str, resolution_note: str = "") -> Optional[dict]:
    """Update the status of an existing ticket."""
    with _lock:
        store = _read(TICKETS_FILE)
        if ticket_id.upper() not in store:
            return None
        store[ticket_id.upper()]["status"]          = status
        store[ticket_id.upper()]["updated_at"]      = _now()
        if resolution_note:
            store[ticket_id.upper()]["resolution_note"] = resolution_note
        _write(TICKETS_FILE, store)
        return store[ticket_id.upper()]


def list_tickets_by_session(session_id: str) -> list[dict]:
    """Return all tickets for a session."""
    with _lock:
        store = _read(TICKETS_FILE)
    return [t for t in store.values() if t["session_id"] == session_id]


def register_account(
    session_id:   str,
    name:         str,
    phone:        str,
    email:        str,
    account_type: str = "savings",
) -> dict:
    """
    Register a new user account for a session.
    If the session already has an account, return the existing one.
    """
    with _lock:
        store = _read(ACCOUNTS_FILE)

        
        for acc in store.values():
            if acc["session_id"] == session_id:
                return acc

        account_number = f"XXXX-XXXX-{uuid.uuid4().hex[:4].upper()}"
        account = {
            "account_id":     str(uuid.uuid4())[:8].upper(),
            "account_number": account_number,
            "session_id":     session_id,
            "name":           name,
            "phone":          phone,
            "email":          email,
            "account_type":   account_type,
            "status":         "Active",
            "opened_at":      _now(),
        }

        store[account["account_id"]] = account
        _write(ACCOUNTS_FILE, store)

    return account


def get_account_by_session(session_id: str) -> Optional[dict]:
    """Return the account linked to this session, or None."""
    with _lock:
        store = _read(ACCOUNTS_FILE)
    for acc in store.values():
        if acc["session_id"] == session_id:
            return acc
    return None


def account_exists(session_id: str) -> bool:
    return get_account_by_session(session_id) is not None

def log_query(
    session_id: str,
    query:      str,
    response:   str,
    intent:     str = "",
    query_type: str = "",
) -> None:
    """Append a query + response pair to the audit log."""
    entry_id = str(uuid.uuid4())[:8]
    entry = {
        "id":         entry_id,
        "session_id": session_id,
        "query":      query,
        "response":   response,
        "intent":     intent,
        "type":       query_type,
        "timestamp":  _now(),
    }

    with _lock:
        store = _read(LOG_FILE)
        if "entries" not in store:
            store["entries"] = []
        store["entries"].append(entry)
        _write(LOG_FILE, store)