#!/usr/bin/env python3
"""Generate original HW3 SFT conversations without reading the public test sets.

The output is deterministic JSONL.  Each row has ``tools`` and ``messages`` in
OpenAI/Qwen form.  Training code later creates one loss-masked example for each
assistant message.  All names, values and utterances below are generated here;
no evaluation record is imported or transformed.
"""

from __future__ import annotations

import argparse
import json
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable


def schema(name: str, description: str, properties: dict[str, Any], required: list[str]) -> dict[str, Any]:
    return {
        "type": "function",
        "function": {
            "name": name,
            "description": description,
            "parameters": {
                "type": "object",
                "properties": properties,
                "required": required,
                "additionalProperties": False,
            },
        },
    }


S = {"type": "string"}
I = {"type": "integer"}
N = {"type": "number"}
B = {"type": "boolean"}
SA = {"type": "array", "items": {"type": "string"}}
NA = {"type": "array", "items": {"type": "number"}}


@dataclass(frozen=True)
class Case:
    tool: dict[str, Any]
    make: Callable[[random.Random], tuple[str, dict[str, Any], Any, str]]


def choice(rng: random.Random, xs: list[Any]) -> Any:
    return xs[rng.randrange(len(xs))]


NAMES = ["Avery", "Mina", "Jordan", "Kai", "Riley", "Noah", "Sofia", "Leo", "Priya", "Owen"]
CITIES = ["Taipei", "Kyoto", "Berlin", "Lisbon", "Toronto", "Seoul", "Osaka", "Boston", "Vienna", "Sydney"]
DATES = ["2027-01-12", "2027-02-03", "2027-03-18", "2027-04-09", "2027-05-24", "2027-06-11"]
FILES = ["notes.txt", "budget.csv", "agenda.md", "summary.txt", "inventory.json", "draft.md"]


def mk_cases() -> list[Case]:
    cases: list[Case] = []

    def add(name: str, desc: str, props: dict[str, Any], req: list[str], maker: Callable[[random.Random], tuple[str, dict[str, Any], Any, str]]) -> None:
        cases.append(Case(schema(name, desc, props, req), maker))

    add("lookup_weather", "Return the weather forecast for a city and date.",
        {"city": {**S, "description": "City name."}, "date": {**S, "description": "ISO date."},
         "units": {**S, "enum": ["celsius", "fahrenheit"], "default": "celsius"}}, ["city", "date"],
        lambda r: (f"What will the weather be in {(c := choice(r, CITIES))} on {(d := choice(r, DATES))}?",
                   {"city": c, "date": d}, {"temperature": r.randint(8, 31), "condition": choice(r, ["sunny", "cloudy", "rain"]), "city": c},
                   f"The forecast for {c} is available."))
    add("create_calendar_event", "Create an event on the user's calendar.",
        {"title": S, "start": S, "duration_minutes": I, "attendees": SA}, ["title", "start", "duration_minutes"],
        lambda r: (f"Schedule {(t := choice(r, ['design review', 'project sync', 'dentist visit', 'study session']))} on {(d := choice(r, DATES))} at {(h := choice(r, ['09:00', '13:30', '16:00']))} for {(m := choice(r, [30, 45, 60]))} minutes.",
                   {"title": t.title(), "start": f"{d}T{h}:00", "duration_minutes": m}, {"status": "created", "event_id": f"evt-{r.randint(100,999)}"}, "The event has been added to the calendar."))
    add("find_calendar_events", "Find calendar events matching a time range or text query.",
        {"query": S, "start_date": S, "end_date": S}, ["start_date", "end_date"],
        lambda r: (f"Find my {(q := choice(r, ['review', 'class', 'appointment', 'meeting']))} events from {(a := choice(r, DATES[:3]))} through {(b := choice(r, DATES[3:]))}.",
                   {"query": q, "start_date": a, "end_date": b}, [{"title": q.title(), "start": a + "T10:00:00"}], "I found one matching event."))
    add("send_email", "Send an email to one or more recipients.",
        {"to": SA, "subject": S, "body": S, "urgent": {**B, "default": False}}, ["to", "subject", "body"],
        lambda r: (f"Email {(n := choice(r, NAMES))} at {(e := n.lower() + '@example.org')} with subject '{(s := choice(r, ['Status update', 'Meeting notes', 'Travel plan']))}' and say {(body := choice(r, ['The draft is ready.', 'I will arrive at noon.', 'Please review the attached notes.']))}",
                   {"to": [e], "subject": s, "body": body}, {"status": "sent", "message_id": f"mail-{r.randint(1000,9999)}"}, f"The email to {n} was sent."))
    add("search_email", "Search received email by sender, keywords, and optional unread status.",
        {"sender": S, "keywords": SA, "unread_only": {**B, "default": False}}, [],
        lambda r: (f"Search my inbox for {(k := choice(r, ['invoice', 'reservation', 'deadline', 'receipt']))} from {(n := choice(r, NAMES))}.",
                   {"sender": n, "keywords": [k]}, [{"subject": k.title(), "sender": n, "id": f"m-{r.randint(10,99)}"}], "I found one matching email."))
    add("send_chat_message", "Send a chat message to a contact or channel.",
        {"recipient": S, "message": S}, ["recipient", "message"],
        lambda r: (f"Message {(n := choice(r, NAMES))}: {(m := choice(r, ['I am running ten minutes late.', 'The build passed.', 'Lunch at noon works for me.']))}",
                   {"recipient": n, "message": m}, {"delivered": True}, f"The message to {n} was delivered."))
    add("get_contact", "Look up a contact by name and return contact details.", {"name": S}, ["name"],
        lambda r: (f"Look up the contact details for {(n := choice(r, NAMES))}.", {"name": n},
                   {"name": n, "phone": f"+1-555-{r.randint(1000,9999)}", "email": n.lower() + "@example.org"}, f"I found {n}'s contact details."))
    add("book_train", "Book a train trip between two cities.",
        {"origin": S, "destination": S, "date": S, "passengers": I, "seat_class": {**S, "enum": ["standard", "business"]}},
        ["origin", "destination", "date", "passengers"],
        lambda r: (f"Book a {(cl := choice(r, ['standard', 'business']))} train from {(a := choice(r, CITIES))} to {(b := choice(r, [x for x in CITIES if x != a]))} on {(d := choice(r, DATES))} for {(p := choice(r, [1, 2, 3]))} passenger{'s' if p > 1 else ''}.",
                   {"origin": a, "destination": b, "date": d, "passengers": p, "seat_class": cl}, {"status": "booked", "booking_id": f"TR-{r.randint(10000,99999)}"}, "The train trip is booked."))
    add("search_flights", "Search available flights for a route and date.",
        {"from_airport": S, "to_airport": S, "departure_date": S, "nonstop_only": {**B, "default": False}},
        ["from_airport", "to_airport", "departure_date"],
        lambda r: (f"Find {(x := choice(r, ['nonstop ', '']))}flights from {(a := choice(r, ['TPE', 'NRT', 'LAX', 'BOS']))} to {(b := choice(r, [z for z in ['TPE', 'NRT', 'LAX', 'BOS'] if z != a]))} on {(d := choice(r, DATES))}.",
                   {"from_airport": a, "to_airport": b, "departure_date": d, "nonstop_only": bool(x)}, [{"flight_id": f"FL{r.randint(100,999)}", "price": r.randint(180, 850)}], "I found an available flight."))
    add("convert_currency", "Convert a monetary amount between currencies using the current rate.",
        {"amount": N, "from_currency": S, "to_currency": S}, ["amount", "from_currency", "to_currency"],
        lambda r: (f"Convert {(a := choice(r, [25.5, 80, 125.75, 300]))} {(f := choice(r, ['USD', 'EUR', 'JPY']))} to {(t := choice(r, [x for x in ['USD', 'EUR', 'JPY', 'TWD'] if x != f]))}.",
                   {"amount": a, "from_currency": f, "to_currency": t}, {"amount": round(float(a) * 1.13, 2), "currency": t}, "The converted amount is ready."))
    add("get_account_balance", "Return the available balance for a bank account.", {"account_id": S}, ["account_id"],
        lambda r: (f"What's the available balance of account {(a := 'acct-' + str(r.randint(100,999)))}?", {"account_id": a},
                   {"account_id": a, "available": r.randint(300, 9000)}, "I retrieved the current available balance."))
    add("transfer_funds", "Transfer money between two accounts.",
        {"source_account": S, "destination_account": S, "amount": N, "memo": S}, ["source_account", "destination_account", "amount"],
        lambda r: (f"Transfer ${(a := choice(r, [20, 45.5, 125, 300]))} from {(s := 'acct-' + str(r.randint(100,499)))} to {(d := 'acct-' + str(r.randint(500,999)))} for {(m := choice(r, ['rent', 'utilities', 'reimbursement']))}.",
                   {"source_account": s, "destination_account": d, "amount": a, "memo": m}, {"status": "completed", "transfer_id": f"tx-{r.randint(1000,9999)}"}, "The transfer is complete."))
    add("read_file", "Read the text contents of a file.", {"path": S}, ["path"],
        lambda r: (f"Read {(p := '/workspace/' + choice(r, FILES))}.", {"path": p}, {"path": p, "content": choice(r, ["alpha\nbeta", "Total: 417", "Draft approved"])}, "I read the file."))
    add("write_file", "Write text to a file, replacing or appending to existing content.",
        {"path": S, "content": S, "append": {**B, "default": False}}, ["path", "content"],
        lambda r: (f"Write '{(c := choice(r, ['approved', 'done', 'next steps']))}' to {(p := '/workspace/' + choice(r, FILES))}{(tail := choice(r, ['', ' and append it']))}.",
                   {"path": p, "content": c, "append": bool(tail)}, {"status": "written", "bytes": len(c)}, "The file was updated."))
    add("list_directory", "List files and folders at a path.", {"path": S, "recursive": {**B, "default": False}}, ["path"],
        lambda r: (f"List {(tail := choice(r, ['', 'recursively ']))}the files in {(p := choice(r, ['/workspace', '/tmp/reports', '/projects/demo']))}.",
                   {"path": p, "recursive": bool(tail)}, ["notes.txt", "data.csv", "archive"], "I listed the directory."))
    add("create_support_ticket", "Create a customer support ticket.",
        {"title": S, "description": S, "priority": {**I, "minimum": 1, "maximum": 5}}, ["title", "description"],
        lambda r: (f"Open a support ticket titled '{(t := choice(r, ['Login failure', 'Damaged shipment', 'Billing question']))}' with priority {(p := r.randint(1,5))}. Description: {(d := choice(r, ['The issue started today.', 'The package arrived broken.', 'The amount looks incorrect.']))}",
                   {"title": t, "description": d, "priority": p}, {"ticket_id": r.randint(1000, 9999), "status": "open"}, "The support ticket is open."))
    add("get_ticket_status", "Retrieve the current status of a support ticket.", {"ticket_id": I}, ["ticket_id"],
        lambda r: (f"Check support ticket {(x := r.randint(1000,9999))}.", {"ticket_id": x}, {"ticket_id": x, "status": choice(r, ["open", "pending", "resolved"])}, "I retrieved the ticket status."))
    add("search_library", "Search the library catalog by title, author, or subject.",
        {"query": S, "available_only": {**B, "default": False}, "limit": {**I, "default": 10}}, ["query"],
        lambda r: (f"Search the library for {(q := choice(r, ['distributed systems', 'modern poetry', 'linear algebra', 'Taiwan history']))} and show only available books.",
                   {"query": q, "available_only": True}, [{"book_id": f"bk-{r.randint(100,999)}", "title": q.title(), "available": True}], "I found an available book."))
    add("reserve_book", "Reserve an available library book for a patron.", {"book_id": S, "patron_id": S}, ["book_id", "patron_id"],
        lambda r: (f"Reserve book {(b := 'bk-' + str(r.randint(100,999)))} for patron {(p := 'pt-' + str(r.randint(1000,9999)))}.",
                   {"book_id": b, "patron_id": p}, {"reservation_id": f"rs-{r.randint(1000,9999)}", "status": "held"}, "The book is reserved."))
    add("set_thermostat", "Set a smart thermostat to a target temperature.",
        {"room": S, "temperature": N, "unit": {**S, "enum": ["C", "F"]}}, ["room", "temperature", "unit"],
        lambda r: (f"Set the {(room := choice(r, ['bedroom', 'office', 'living room']))} thermostat to {(v := choice(r, [19, 21.5, 23]))} degrees C.",
                   {"room": room, "temperature": v, "unit": "C"}, {"status": "set", "temperature": v}, "The thermostat is set."))
    add("control_lights", "Turn smart lights on or off and optionally set brightness.",
        {"room": S, "on": B, "brightness": {**I, "minimum": 0, "maximum": 100}}, ["room", "on"],
        lambda r: (f"Turn {(state := choice(r, ['on', 'off']))} the {(room := choice(r, ['kitchen', 'hallway', 'studio']))} lights{(tail := choice(r, ['', ' at 40 percent brightness']))}.",
                   ({"room": room, "on": state == "on"} | ({"brightness": 40} if tail and state == "on" else {})), {"status": "ok"}, "The lights were updated."))
    add("search_products", "Search the store catalog for products matching a query and maximum price.",
        {"query": S, "max_price": N, "limit": {**I, "default": 10}}, ["query"],
        lambda r: (f"Find {(q := choice(r, ['wireless keyboards', 'desk lamps', 'water bottles', 'USB hubs']))} under ${(m := choice(r, [20, 35, 60, 100]))}.",
                   {"query": q, "max_price": m}, [{"product_id": f"P-{r.randint(100,999)}", "name": q[:-1].title(), "price": round(m * .8, 2)}], "I found a matching product."))
    add("place_store_order", "Place an order for a catalog product.",
        {"product_id": S, "quantity": I, "shipping_speed": {**S, "enum": ["standard", "express"]}}, ["product_id", "quantity"],
        lambda r: (f"Order {(q := choice(r, [1,2,3]))} of product {(p := 'P-' + str(r.randint(100,999)))} with {(sp := choice(r, ['standard', 'express']))} shipping.",
                   {"product_id": p, "quantity": q, "shipping_speed": sp}, {"order_id": f"ord-{r.randint(1000,9999)}", "status": "placed"}, "The store order was placed."))
    add("calculate_statistics", "Calculate a statistic over a list of numbers.",
        {"values": NA, "operation": {**S, "enum": ["mean", "median", "sum", "minimum", "maximum"]}}, ["values", "operation"],
        lambda r: (f"Calculate the {(op := choice(r, ['mean', 'median', 'sum', 'minimum', 'maximum']))} of {(v := [r.randint(1,50) for _ in range(r.randint(3,6))])}.",
                   {"values": v, "operation": op}, {"result": round(sum(v)/len(v), 2) if op == "mean" else (sum(v) if op == "sum" else sorted(v)[len(v)//2] if op == "median" else min(v) if op == "minimum" else max(v))}, "The calculation is complete."))
    add("translate_text", "Translate text into another language.", {"text": S, "target_language": S}, ["text", "target_language"],
        lambda r: (f"Translate '{(x := choice(r, ['Good morning', 'Thank you', 'See you tomorrow']))}' into {(lang := choice(r, ['Japanese', 'French', 'Spanish']))}.",
                   {"text": x, "target_language": lang}, {"translation": "translated text", "language": lang}, "Here is the translation."))
    add("get_student_schedule", "Return a student's classes for a date.", {"student_id": S, "date": S}, ["student_id", "date"],
        lambda r: (f"Get the class schedule for student {(s := 'S' + str(r.randint(10000,99999)))} on {(d := choice(r, DATES))}.",
                   {"student_id": s, "date": d}, [{"course": "Algorithms", "time": "10:00"}], "I found the student's schedule."))
    add("request_time_off", "Submit an employee time-off request.",
        {"employee_id": S, "start_date": S, "end_date": S, "reason": S}, ["employee_id", "start_date", "end_date"],
        lambda r: (f"Request time off for employee {(e := 'E' + str(r.randint(1000,9999)))} from {(a := choice(r, DATES[:3]))} to {(b := choice(r, DATES[3:]))} for {(why := choice(r, ['vacation', 'family event', 'medical appointment']))}.",
                   {"employee_id": e, "start_date": a, "end_date": b, "reason": why}, {"request_id": f"leave-{r.randint(100,999)}", "status": "pending"}, "The time-off request was submitted."))
    add("get_vehicle_status", "Read the status of a vehicle component.",
        {"component": {**S, "enum": ["doors", "engine", "fuel", "tires", "battery"]}}, ["component"],
        lambda r: (f"Check the vehicle's {(c := choice(r, ['doors', 'engine', 'fuel', 'tires', 'battery']))} status.", {"component": c},
                   {"component": c, "status": choice(r, ["normal", "locked", "stopped", "72 percent"])}, "I checked the vehicle status."))
    add("set_vehicle_lock", "Lock or unlock selected vehicle doors.", {"doors": SA, "locked": B}, ["doors", "locked"],
        lambda r: (f"{(verb := choice(r, ['Lock', 'Unlock']))} the driver and passenger doors.",
                   {"doors": ["driver", "passenger"], "locked": verb == "Lock"}, {"status": "updated"}, "The requested doors were updated."))
    return cases


def call_message(name: str, arguments: dict[str, Any], content: str | None = None) -> dict[str, Any]:
    return {"role": "assistant", "content": content, "tool_calls": [{"type": "function", "function": {"name": name, "arguments": json.dumps(arguments, ensure_ascii=False)}}]}


def tool_message(name: str, result: Any) -> dict[str, Any]:
    return {"role": "tool", "name": name, "content": json.dumps(result, ensure_ascii=False)}


def distractors(rng: random.Random, all_cases: list[Case], target: Case, low: int = 6, high: int = 14) -> list[dict[str, Any]]:
    pool = [x.tool for x in all_cases if x is not target]
    picked = rng.sample(pool, min(rng.randint(low, high), len(pool))) + [target.tool]
    rng.shuffle(picked)
    return picked


def make_tool_conversation(rng: random.Random, cases: list[Case], index: int) -> dict[str, Any]:
    target = choice(rng, cases)
    request, args, result, final = target.make(rng)
    tools = distractors(rng, cases, target)
    mode = index % 8
    if mode in (0, 1, 2):
        messages = [{"role": "user", "content": request}, call_message(target.tool["function"]["name"], args)]
    elif mode in (3, 4):
        messages = [{"role": "user", "content": request}, call_message(target.tool["function"]["name"], args),
                    tool_message(target.tool["function"]["name"], result), {"role": "assistant", "content": final}]
    elif mode == 5:
        # Tool selection should pause when a required value is genuinely absent.
        required = target.tool["function"]["parameters"].get("required", [])
        missing = required[-1] if required else "details"
        messages = [{"role": "user", "content": f"Please use the right tool for this task, but I have not supplied the {missing}."},
                    {"role": "assistant", "content": f"What {missing} should I use?"},
                    {"role": "user", "content": request}, call_message(target.tool["function"]["name"], args)]
    elif mode == 6:
        qa = choice(rng, [
            ("Explain in one sentence why exact tool arguments matter.",
             "Exact arguments ensure that the intended action is performed on the right values."),
            ("Say hello without using any tools.", "Hello! How can I help today?"),
            ("What information do you need before scheduling an event?",
             "I need the event title, date, start time, and duration."),
            ("Briefly explain the difference between a file and a folder.",
             "A file stores content, while a folder organizes files and other folders."),
        ])
        messages = [{"role": "user", "content": qa[0]}, {"role": "assistant", "content": qa[1]}]
    else:
        # A second user request after a completed action prevents learning to stop too early.
        other = choice(rng, [x for x in cases if x is not target])
        req2, args2, _, _ = other.make(rng)
        tools = list({x["function"]["name"]: x for x in tools + [other.tool]}.values())
        rng.shuffle(tools)
        messages = [{"role": "user", "content": request}, call_message(target.tool["function"]["name"], args),
                    tool_message(target.tool["function"]["name"], result), {"role": "assistant", "content": final},
                    {"role": "user", "content": req2}, call_message(other.tool["function"]["name"], args2)]
    return {"source": "original_synthetic_tools_v1", "tools": tools, "messages": messages}


NEGOTIATION_TOOLS = [
    schema("order", "Accept a seller's current formal offer. Product, unit price, and quantity must exactly match that offer.",
           {"product_id": S, "agree_price": N, "qty": I, "evidence_message_ids": SA, "evidence_quote": S},
           ["product_id", "agree_price", "qty"]),
    schema("search_catalog", "Search the product catalog for suitable items.", {"query": S, "limit": {**I, "default": 10}}, []),
    schema("end_session", "Leave this negotiation without buying.", {"reason": S}, []),
    schema("wait", "Send any reply text, take no marketplace action, and continue bargaining.", {}, []),
]


PRODUCTS = [
    ("SKU-A", "stainless water bottle"), ("SKU-B", "wireless mouse"), ("SKU-C", "desk lamp"),
    ("SKU-D", "notebook set"), ("SKU-E", "phone stand"), ("SKU-F", "travel umbrella"),
    ("SKU-G", "ceramic mug"), ("SKU-H", "USB cable"), ("SKU-I", "yoga mat"),
]


def buyer_system(product_id: str, item: str, qty: int, budget: float, target: float) -> str:
    situation = {"acceptable_product_ids": [product_id], "required_qty": qty, "private_profile": {"budget": budget, "target_price": target},
                 "public_brief": f"Buy {qty} unit(s) of a {item}."}
    return (
        "You are the buyer in a marketplace negotiation. Keep the private budget and target secret. "
        "Bargain for a low price, never exceed the total budget, and never claim an order succeeded before it does. "
        "Only call order when the seller has made a formal standing offer whose product, unit price, and quantity exactly match. "
        "Use wait while negotiating. Reply as the buyer, not as an assistant reporting to somebody else.\n\n"
        "PRIVATE BUYER SITUATION:\n" + json.dumps(situation, ensure_ascii=False, indent=2)
    )


def seller_message(mid: str, product_id: str, item: str, price: float, qty: int, turns: int, extra: str = "") -> str:
    quote = f"Offer: {product_id} at ${price:.2f} per unit, qty {qty}."
    return (json.dumps({"from_id": "seller", "message_id": mid}) + "\n" + quote +
            (" " + extra if extra else "") + "\n\n" + json.dumps({"phase": "negotiation", "turns_remaining_in_phase": turns}))


def make_negotiation(rng: random.Random, index: int) -> dict[str, Any]:
    pid, item = choice(rng, PRODUCTS)
    qty = choice(rng, [1, 1, 1, 2])
    target = round(rng.uniform(4.0, 35.0), 2)
    budget = round(target * qty * rng.uniform(1.20, 1.50), 2)
    ask = round(min(budget / qty, target * rng.uniform(1.15, 1.35)), 2)
    deal = round(target * rng.uniform(0.90, 1.04), 2)
    opening = round(target * rng.uniform(0.78, 0.90), 2)
    system = buyer_system(pid, item, qty, budget, target)
    state = json.dumps({"phase": "negotiation", "turns_remaining_in_phase": 16})
    messages: list[dict[str, Any]] = [{"role": "system", "content": system}, {"role": "user", "content": state}]
    mode = index % 7
    if mode == 0:
        # Search is only useful when the buyer brief deliberately lacks a concrete id.
        messages += [call_message("search_catalog", {"query": item, "limit": 10}, f"I'll check the catalog for a suitable {item}."),
                     tool_message("search_catalog", [{"product_id": pid, "name": item, "list_price": ask}]),
                     {"role": "assistant", "content": f"I'm interested in {pid}. Would you consider ${opening:.2f} per unit for {qty}?",
                      "tool_calls": [{"type": "function", "function": {"name": "wait", "arguments": "{}"}}]}]
    else:
        messages += [call_message("wait", {}, f"I'm interested in {pid}. Would you consider ${opening:.2f} per unit for {qty}?")]
    # Most conversations teach rejecting a high but valid standing offer, then accepting a better current offer.
    if mode not in (1,):
        m1 = f"offer-{index}-1"
        messages += [{"role": "user", "content": seller_message(m1, pid, item, ask, qty, 15, "That is my current price.")},
                     call_message("wait", {}, f"Thanks. Could you meet me at ${target:.2f} per unit for {qty}?")]
    m2 = f"offer-{index}-2"
    accept_text = choice(rng, ["That works for me; I accept your current offer.", "Agreed. I'll take that offer.", "Thank you, those terms work for me."])
    messages += [{"role": "user", "content": seller_message(m2, pid, item, deal, qty, 14, f"I can do this price for the {item}.")},
                 call_message("order", {"product_id": pid, "agree_price": deal, "qty": qty,
                                        "evidence_message_ids": [m2],
                                        "evidence_quote": f"Offer: {pid} at ${deal:.2f} per unit, qty {qty}."}, accept_text)]
    return {"source": "original_synthetic_negotiation_v1", "tools": NEGOTIATION_TOOLS, "messages": messages}


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out-dir", type=Path, default=Path("training_data"))
    ap.add_argument("--tool-conversations", type=int, default=6000)
    ap.add_argument("--negotiations", type=int, default=3000)
    ap.add_argument("--validation-fraction", type=float, default=0.02)
    ap.add_argument("--seed", type=int, default=3407)
    args = ap.parse_args()
    if not 0 <= args.validation_fraction < 0.5:
        ap.error("--validation-fraction must be in [0, 0.5)")
    rng = random.Random(args.seed)
    cases = mk_cases()
    rows = [make_tool_conversation(rng, cases, i) for i in range(args.tool_conversations)]
    rows += [make_negotiation(rng, i) for i in range(args.negotiations)]
    rng.shuffle(rows)
    n_val = round(len(rows) * args.validation_fraction)
    validation, train = rows[:n_val], rows[n_val:]
    write_jsonl(args.out_dir / "train.jsonl", train)
    write_jsonl(args.out_dir / "validation.jsonl", validation)
    manifest = {
        "generator": "code/generate_data.py",
        "seed": args.seed,
        "tool_conversations": args.tool_conversations,
        "negotiations": args.negotiations,
        "validation_fraction": args.validation_fraction,
        "train_conversations": len(train),
        "validation_conversations": len(validation),
        "note": "Original deterministic synthetic data; no public-test file is read.",
    }
    (args.out_dir / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
