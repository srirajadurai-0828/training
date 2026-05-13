# ─── record_tools.py ─────────────────────────────────────────────────────────

import json
from langchain_core.tools import tool
from storage.data_store import (
    create_ticket,
    get_ticket,
    get_account_by_session,
    list_tickets_by_session,
)


@tool
def raise_complaint_ticket(
    session_id:          str,
    query:               str,
    complaint_type:      str,
    priority:            str,
    monetary_impact:     str,
    escalation_required: bool,
    sla_hours:           int,
    reason:              str,
    amount:              str = "",
    transaction_date:    str = "",
) -> str:
    """
    Create a complaint or dispute ticket and save it to the database.

    Use this whenever the user reports:
    - A duplicate charge or wrong debit
    - A failed transaction where money was deducted
    - Fraud or unauthorised activity
    - Any unresolved banking complaint

    Args:
        session_id:          The current session ID (always pass this).
        query:               The user's original complaint text.
        complaint_type:      Short label e.g. 'duplicate_charge', 'failed_transfer', 'fraud'.
        priority:            P1 (critical/fraud) | P2 (money issue) | P3 (service) | P4 (feedback).
        monetary_impact:     'Very High' | 'High' | 'Medium' | 'Low' | 'None'.
        escalation_required: True if this needs human review.
        sla_hours:           Expected resolution time in hours (1, 4, 24, or 48).
        reason:              One-line reason for the priority assigned.
        amount:              Transaction amount involved, if known (e.g. '₹2,340').
        transaction_date:    Date of the disputed transaction, if known (e.g. '12-Jan-2025').

    Returns:
        A confirmation string with the ticket ID, priority, and SLA.
    """
    ticket = create_ticket(
        session_id          = session_id,
        query               = query,
        complaint_type      = complaint_type,
        priority            = priority,
        monetary_impact     = monetary_impact,
        escalation_required = escalation_required,
        sla_hours           = sla_hours,
        reason              = reason,
        amount              = amount or None,
        date                = transaction_date or None,
    )

    sla_text = (
        f"{ticket['sla_hours']} hour"
        if ticket["sla_hours"] == 1
        else f"{ticket['sla_hours']} hours"
    )

    lines = [
        f"Ticket raised successfully.",
        f"Ticket ID:    {ticket['ticket_id']}",
        f"Priority:     {ticket['priority']} — {ticket['reason']}",
        f"Status:       {ticket['status']}",
        f"SLA:          Resolution expected within {sla_text}",
    ]
    if ticket.get("amount"):
        lines.append(f"Amount:       {ticket['amount']}")
    if ticket.get("transaction_date"):
        lines.append(f"Txn Date:     {ticket['transaction_date']}")
    if ticket["escalation_required"]:
        lines.append("Escalation:   Yes — flagged for human review")

    return "\n".join(lines)


@tool
def check_ticket_status(ticket_id: str) -> str:
    """
    Look up the status and details of a complaint or dispute ticket.

    Use this when the user asks about the status of an existing ticket,
    e.g. 'What is the status of ticket DC-4A3F?'

    Args:
        ticket_id: The ticket ID (e.g. DC-4A3F, TKT-9B2E, EMG-1C4D).

    Returns:
        Full ticket details as a formatted string, or a not-found message.
    """
    ticket = get_ticket(ticket_id.upper().strip())

    if not ticket:
        return (
            f"No ticket found with ID '{ticket_id}'. "
            "Please double-check the ticket ID. "
            "If you raised the ticket in a different session, contact the helpline."
        )

    lines = [
        f"Ticket ID:    {ticket['ticket_id']}",
        f"Status:       {ticket['status']}",
        f"Priority:     {ticket['priority']}",
        f"Type:         {ticket.get('complaint_type', '—')}",
        f"Reason:       {ticket.get('reason', '—')}",
        f"Raised on:    {ticket['created_at']}",
        f"Last updated: {ticket['updated_at']}",
        f"SLA:          {ticket['sla_hours']} hours",
    ]
    if ticket.get("amount"):
        lines.append(f"Amount:       {ticket['amount']}")
    if ticket.get("transaction_date"):
        lines.append(f"Txn Date:     {ticket['transaction_date']}")
    if ticket.get("resolution_note"):
        lines.append(f"Resolution:   {ticket['resolution_note']}")
    if ticket.get("escalation_required"):
        lines.append("Escalation:   Flagged for human review")

    return "\n".join(lines)


@tool
def list_my_tickets(session_id: str) -> str:
    """
    List all tickets raised in the current session.

    Use this when the user asks 'show my tickets' or 'what complaints have I raised?'

    Args:
        session_id: The current session ID.

    Returns:
        A formatted list of all tickets for the session.
    """
    tickets = list_tickets_by_session(session_id)

    if not tickets:
        return "No tickets found for this session."

    lines = [f"Found {len(tickets)} ticket(s) for your session:\n"]
    for t in tickets:
        lines.append(
            f"• {t['ticket_id']}  [{t['priority']}]  {t['status']}  —  {t.get('reason','')}"
            f"  (raised {t['created_at']})"
        )

    return "\n".join(lines)


@tool
def check_account_status(session_id: str) -> str:
    """
    Check whether this session has a registered account and return its details.

    ALWAYS call this tool first when the user asks about:
    - Their account balance
    - Their account details
    - Any account-specific service

    Args:
        session_id: The current session ID.

    Returns:
        Account summary if found, or a message that no account is linked.
    """
    account = get_account_by_session(session_id)

    if not account:
        return (
            "NO_ACCOUNT: No account is linked to this session. "
            "Please visit your nearest branch or contact the helpline "
            "to open an account or link an existing one."
        )

    return (
        f"Account found.\n"
        f"Name:           {account['name']}\n"
        f"Account No:     {account['account_number']}\n"
        f"Type:           {account['account_type'].title()}\n"
        f"Status:         {account['status']}\n"
        f"Opened:         {account['opened_at']}"
    )