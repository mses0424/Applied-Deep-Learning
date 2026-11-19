#!/usr/bin/env python3
"""Independent policy simulations with explicit prerequisites and state changes.

This generator uses only locally authored service vocabulary and rules.  It
never reads evaluation records.  Every tool result is computed from the state
chosen for that episode, and every later argument is copied from that state.
"""

from __future__ import annotations

import argparse
import collections
import hashlib
import json
import random
from pathlib import Path
from typing import Any

from generate_policy_v10 import DOMAINS, Domain, call, ids, result, schema, toolset


MORE_DOMAINS = [
    Domain("aviation_parts", "aviation parts cooperative", "mechanic", "replacement assembly", "parts request"),
    Domain("garden_lending", "community garden lending library", "gardener", "shared planter", "planter loan"),
    Domain("marine_samples", "marine sampling center", "researcher", "sample container", "sampling reservation"),
    Domain("theater_wardrobe", "theater wardrobe office", "cast member", "wardrobe set", "wardrobe loan"),
    Domain("campus_labs", "campus laboratory service", "lab member", "analysis station", "lab booking"),
    Domain("sports_facility", "sports training facility", "club member", "training court", "court reservation"),
    Domain("geology_store", "geology sample store", "field scientist", "sample drawer", "sample access request"),
    Domain("water_testing", "water quality testing office", "client", "test apparatus", "testing appointment"),
    Domain("music_rehearsal", "music rehearsal cooperative", "performer", "rehearsal suite", "suite booking"),
    Domain("maker_inventory", "maker-space inventory desk", "fabricator", "tool locker", "locker loan"),
    Domain("research_greenhouse", "research greenhouse office", "investigator", "growth chamber", "chamber allocation"),
    Domain("mobile_exhibit", "traveling exhibition office", "curator", "display case", "exhibit loan"),
    Domain("science_outreach", "science outreach center", "educator", "demonstration kit", "kit reservation"),
    Domain("trail_service", "public trail service", "volunteer", "maintenance vehicle", "vehicle booking"),
    Domain("archive_scanning", "archive digitization office", "archivist", "scanning station", "scanning slot"),
    Domain("textile_lab", "textile testing laboratory", "designer", "weaving bench", "bench booking"),
    Domain("agri_sensor", "agricultural sensor pool", "technician", "sensor array", "array loan"),
    Domain("lighting_pool", "stage lighting equipment pool", "producer", "lighting rig", "rig loan"),
    Domain("ceramics_studio", "ceramics studio desk", "artist", "firing kiln", "kiln reservation"),
    Domain("seismology_pool", "seismology equipment pool", "investigator", "recording unit", "unit booking"),
    Domain("urban_forest", "urban forest program", "volunteer", "survey plot", "plot reservation"),
    Domain("robot_parts", "robotics components store", "builder", "controller kit", "kit request"),
    Domain("accessibility_lab", "accessibility equipment center", "client", "assistive device", "device loan"),
    Domain("solar_monitor", "solar monitoring center", "technician", "metering unit", "meter reservation"),
]


def assistant_text(body: str) -> dict[str, str]:
    return {"role": "assistant", "content": body}


def validate_case(row: dict[str, Any]) -> None:
    """Check calls against the locally generated schemas and result order."""
    tools = {tool["function"]["name"]: tool["function"]["parameters"] for tool in row["tools"]}
    if len(tools) != len(row["tools"]):
        raise ValueError("duplicate tool names")
    messages = row["messages"]
    for index, message in enumerate(messages):
        if message.get("role") == "assistant" and message.get("tool_calls"):
            for item in message["tool_calls"]:
                function = item["function"]
                name, arguments = function["name"], function["arguments"]
                if name not in tools or not isinstance(arguments, dict):
                    raise ValueError("unknown tool or non-object arguments")
                definition = tools[name]
                if not set(definition.get("required", [])) <= set(arguments):
                    raise ValueError(f"missing required argument in {name}")
                if not set(arguments) <= set(definition.get("properties", {})):
                    raise ValueError(f"unexpected argument in {name}")
                for key, value in arguments.items():
                    field = definition["properties"][key]
                    if field.get("type") == "string" and not isinstance(value, str):
                        raise ValueError(f"non-string argument {name}.{key}")
                    if "enum" in field and value not in field["enum"]:
                        raise ValueError(f"out-of-enum argument {name}.{key}")
                if index + 1 >= len(messages) or messages[index + 1].get("role") != "tool":
                    raise ValueError(f"missing result after {name}")
                if messages[index + 1].get("name") != name:
                    raise ValueError(f"result name mismatch after {name}")
                json.loads(messages[index + 1]["content"])


def build_case(domain: Domain, number: int, family: int, rng: random.Random) -> dict[str, Any]:
    made = toolset(domain)
    names = made["names"]
    keys = ids(domain, number)
    location = rng.choice(["East Campus", "River Annex", "West Depot", "Central Hub"])
    date = rng.choice(["2027-03-16", "2027-04-23", "2027-05-07", "2027-06-12"])
    next_date = rng.choice(["2027-07-08", "2027-08-19", "2027-09-14"])
    slot = rng.choice(["10:00", "13:30", "15:00"])
    email = f"member{number}@example.org"
    audit_id = f"AUD-{number:06d}"
    policy_tool = f"get_{domain.slug}_policy"
    audit_tool = f"get_{domain.slug}_audit"
    tools = [json.loads(json.dumps(tool)) for tool in made["tools"]]
    tools.extend([
        schema(policy_tool, "Read current booking limits and requirements for a service location.",
               {"location": {"type": "string"}}, ["location"]),
        schema(audit_tool, "Check the latest compliance audit of an account before an urgent booking.",
               {"account_id": {"type": "string"}}, ["account_id"]),
    ])
    # The policy explicitly calls for these values in the corresponding
    # branches.  They are optional in the API, which tests whether the model
    # reads policy instead of blindly supplying only required parameters.
    check_props = tools[4]["function"]["parameters"]["properties"]
    check_props["audit_id"] = {"type": "string", "description": "Latest audit reference when urgent service is requested."}
    create_props = tools[5]["function"]["parameters"]["properties"]
    create_props["status"] = {"type": "string", "enum": ["draft", "confirmed"],
                              "description": "Set to confirmed when an immediate reservation is requested."}
    system = (
        f"You assist the {domain.organization}. Never invent a database value. "
        "Read the current policy before a new request. Verify account status, "
        "find an available item, then check eligibility before creating it. "
        "For an urgent request, read the account audit and send its audit_id to "
        "the eligibility check. Immediate reservations must use status=confirmed. "
        "Before changing a request, read its current record. Never cancel without "
        "explicit confirmation. Ask for missing date or account details. "
        "Stop when a tool reports denial or the requested work is complete."
    )
    messages: list[dict[str, Any]] = [{"role": "system", "content": system}]

    def add_call(key: str, arguments: dict[str, Any], payload: dict[str, Any]) -> None:
        messages.append(call(names[key], arguments))
        messages.append(result(names[key], payload))

    def add_special(name: str, arguments: dict[str, Any], payload: dict[str, Any]) -> None:
        messages.append(call(name, arguments))
        messages.append(result(name, payload))

    if family in (0, 1, 4, 6):
        urgent = family in (1, 4)
        messages.append({"role": "user", "content": (
            f"Please {'urgently ' if urgent else ''}reserve a {domain.item} for account {keys['account']} "
            f"at {location} on {date}; confirm it for {slot}."
        )})
        add_special(policy_tool, {"location": location},
                    {"location": location, "must_verify_account": True,
                     "must_check_eligibility": True, "urgent_requires_audit": True,
                     "immediate_status": "confirmed"})
        add_call("get_account", {"account_id": keys["account"]},
                 {"account_id": keys["account"], "status": "active", "request_quota": 2})
        if urgent:
            add_special(audit_tool, {"account_id": keys["account"]},
                        {"account_id": keys["account"], "audit_id": audit_id,
                         "approved": family == 1})
        if family == 4:
            messages.append(assistant_text("The latest audit has not approved this account, so I cannot reserve the item."))
        else:
            add_call("list_items", {"location": location, "date": date},
                     {"location": location, "date": date,
                      "available": [{"item_id": keys["item"], "slot": slot}] if family != 6 else []})
            if family == 6:
                messages.append(assistant_text(f"No {domain.item} is available at {location} on {date}."))
            else:
                check_args = {"account_id": keys["account"], "item_id": keys["item"], "date": date}
                if urgent:
                    check_args["audit_id"] = audit_id
                add_call("check", check_args,
                         {"eligible": True, "eligibility_token": keys["token"],
                          "account_id": keys["account"], "item_id": keys["item"]})
                create_args = {"account_id": keys["account"], "item_id": keys["item"],
                               "date": date, "slot": slot,
                               "eligibility_token": keys["token"], "status": "confirmed"}
                add_call("create", create_args,
                         {"request_id": keys["request"], "status": "confirmed", "slot": slot})
                messages.append(assistant_text(f"Reservation {keys['request']} is confirmed for {date} at {slot}."))
    elif family == 2:
        messages.append({"role": "user", "content": f"Move my existing {domain.transaction} to {next_date} at {slot}. My email is {email}."})
        add_call("find_account", {"email": email}, {"email": email, "account_id": keys["account"]})
        add_call("history", {"account_id": keys["account"]},
                 {"records": [{"request_id": keys["request"], "status": "confirmed"}]})
        add_call("get_txn", {"request_id": keys["request"]},
                 {"request_id": keys["request"], "owner_account_id": keys["account"], "can_update": True})
        add_call("update", {"request_id": keys["request"], "date": next_date, "slot": slot},
                 {"request_id": keys["request"], "date": next_date, "slot": slot, "status": "confirmed"})
        messages.append(assistant_text(f"Request {keys['request']} is moved to {next_date} at {slot}."))
    elif family == 3:
        messages.append({"role": "user", "content": f"Cancel request {keys['request']} right away."})
        messages.append(assistant_text(f"Please confirm the cancellation of {keys['request']} and tell me why."))
        messages.append({"role": "user", "content": "Yes, cancel it because our plans changed."})
        add_call("get_txn", {"request_id": keys["request"]},
                 {"request_id": keys["request"], "status": "confirmed", "can_cancel": True})
        add_call("cancel", {"request_id": keys["request"], "reason": "plans changed"},
                 {"request_id": keys["request"], "status": "cancelled"})
        messages.append(assistant_text(f"Request {keys['request']} has been cancelled."))
    elif family == 5:
        messages.append({"role": "user", "content": f"Reserve a {domain.item} for account {keys['account']} at {location}."})
        messages.append(assistant_text("Which date would you like to reserve?"))
        messages.append({"role": "user", "content": f"Use {date}."})
        add_special(policy_tool, {"location": location},
                    {"location": location, "must_verify_account": True,
                     "must_check_eligibility": True, "immediate_status": "confirmed"})
        add_call("get_account", {"account_id": keys["account"]},
                 {"account_id": keys["account"], "status": "suspended", "reason": "expired credentials"})
        messages.append(assistant_text("This account is suspended because its credentials expired; I cannot reserve the item."))
    else:
        messages.append({"role": "user", "content": "Can you publish an advertisement in a local newspaper for me?"})
        messages.append(assistant_text("I can help with service requests, but I cannot publish newspaper advertisements using these tools."))

    row = {"source": f"policy_v11_{family}",
           "source_group": f"policy-v11-{domain.slug}-{number}",
           "tools": tools, "messages": messages,
           "truncate_history_to_last_user": False}
    validate_case(row)
    return row


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out-dir", type=Path, default=Path("training_data_policy_v11"))
    parser.add_argument("--per-domain", type=int, default=104)
    parser.add_argument("--validation-per-domain", type=int, default=8)
    parser.add_argument("--seed", type=int, default=20261027)
    args = parser.parse_args()
    if args.validation_per_domain >= args.per_domain or args.per_domain < 8:
        raise SystemExit("Need at least eight episodes per domain and a smaller validation size")
    rng = random.Random(args.seed)
    train, validation = [], []
    source_counts: collections.Counter[str] = collections.Counter()
    domains = DOMAINS + MORE_DOMAINS
    for index, domain in enumerate(domains):
        order = list(range(args.per_domain))
        rng.shuffle(order)
        held_out = set(order[:args.validation_per_domain])
        for local in range(args.per_domain):
            row = build_case(domain, index * 10000 + local, local % 8, rng)
            source_counts[row["source"]] += 1
            (validation if local in held_out else train).append(row)
    rng.shuffle(train)
    rng.shuffle(validation)
    args.out_dir.mkdir(parents=True, exist_ok=True)
    for name, rows in (("train", train), ("validation", validation)):
        with (args.out_dir / f"{name}.jsonl").open("w", encoding="utf-8") as handle:
            for row in rows:
                handle.write(json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n")
    manifest = {"generator": "code/generate_policy_v11.py", "seed": args.seed,
                "sources": "Locally authored service vocabulary and deterministic stateful policy simulator",
                "domains": len(domains), "train_conversations": len(train),
                "validation_conversations": len(validation),
                "source_counts": dict(sorted(source_counts.items())),
                "public_test_usage": "none"}
    (args.out_dir / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
