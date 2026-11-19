#!/usr/bin/env python3
"""Generate original policy-driven tool-use episodes for HW3 v10.

This generator is deliberately self-contained.  It does not read either HW3
test set (or any other conversation corpus).  Every tool result is produced
from the synthetic state used to construct the episode, so later calls and
the closing response are grounded in information visible in the history.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class Domain:
    slug: str
    organization: str
    account: str
    item: str
    transaction: str


# These independently chosen domains are generic service settings.  The list
# is fixed in source so generation is reproducible and cannot depend on the
# homework evaluation set.
DOMAINS = [
    Domain("lab_supply", "laboratory supply cooperative", "researcher", "instrument kit", "equipment request"),
    Domain("film_archive", "regional film archive", "research visitor", "screening copy", "viewing request"),
    Domain("orchard_service", "orchard equipment service", "grower", "field machine", "service booking"),
    Domain("freight_depot", "freight consolidation depot", "shipping client", "loading bay", "depot booking"),
    Domain("language_center", "language practice center", "learner", "practice room", "room session"),
    Domain("conference_ops", "conference operations desk", "presenter", "presentation room", "session booking"),
    Domain("energy_audit", "home energy audit program", "resident", "audit slot", "audit appointment"),
    Domain("repair_collective", "community repair collective", "member", "repair bench", "bench reservation"),
    Domain("data_vault", "research data vault", "depositor", "storage allocation", "deposit request"),
    Domain("marina_service", "small-craft marina service", "vessel owner", "service berth", "berth booking"),
    Domain("media_studio", "independent media studio", "producer", "recording suite", "studio reservation"),
    Domain("materials_exchange", "reusable materials exchange", "participant", "material lot", "pickup request"),
    Domain("robotics_hub", "robotics prototyping hub", "builder", "test platform", "platform booking"),
    Domain("translation_desk", "document translation desk", "requester", "review slot", "review request"),
    Domain("field_station", "ecology field station", "investigator", "sampling zone", "field permit"),
    Domain("costume_store", "performing-arts costume store", "production member", "costume set", "loan request"),
    Domain("cold_storage", "shared cold-storage facility", "food producer", "storage compartment", "storage booking"),
    Domain("heritage_lab", "heritage conservation laboratory", "collection steward", "treatment station", "treatment request"),
    Domain("print_workshop", "specialty print workshop", "artist", "printing press", "press booking"),
    Domain("field_vehicle", "field vehicle pool", "project member", "utility vehicle", "vehicle reservation"),
    Domain("testing_center", "product testing center", "manufacturer", "test chamber", "test booking"),
    Domain("seed_exchange", "regional seed exchange", "participant", "seed lot", "exchange request"),
    Domain("audio_library", "audio preservation library", "archive user", "listening station", "listening booking"),
    Domain("fabrication_desk", "small-batch fabrication desk", "designer", "fabrication cell", "production request"),
]


def safe(text: str) -> str:
    return "_".join(text.lower().replace("-", " ").split())


def schema(name: str, description: str, properties: dict[str, Any], required: list[str]) -> dict[str, Any]:
    return {
        "type": "function",
        "function": {
            "name": name,
            "description": description,
            "parameters": {"type": "object", "properties": properties, "required": required},
        },
    }


def string(description: str) -> dict[str, str]:
    return {"type": "string", "description": description}


def toolset(domain: Domain) -> dict[str, Any]:
    a, item, txn = safe(domain.account), safe(domain.item), safe(domain.transaction)
    names = {
        "find_account": f"find_{a}_account",
        "get_account": f"get_{a}_profile",
        "list_items": f"list_available_{item}s",
        "get_item": f"get_{item}_details",
        "check": f"check_{txn}_eligibility",
        "create": f"create_{txn}",
        "get_txn": f"get_{txn}",
        "update": f"update_{txn}",
        "cancel": f"cancel_{txn}",
        "history": f"list_{txn}_history",
        "notice": f"send_{txn}_notice",
        "hours": f"get_{domain.slug}_service_hours",
    }
    tools = [
        schema(names["find_account"], f"Find a {domain.account} account by email when its account ID is unknown.",
               {"email": string("Exact email address supplied by the user.")}, ["email"]),
        schema(names["get_account"], f"Retrieve the current {domain.account} profile and account status.",
               {"account_id": string(f"Identifier of the {domain.account} account.")}, ["account_id"]),
        schema(names["list_items"], f"List available {domain.item}s for a location and date.",
               {"location": string("Requested service location."), "date": string("Requested date in YYYY-MM-DD form.")},
               ["location", "date"]),
        schema(names["get_item"], f"Retrieve restrictions and details for one {domain.item}.",
               {"item_id": string(f"Identifier of the {domain.item}.")}, ["item_id"]),
        schema(names["check"], f"Check policy eligibility before creating a {domain.transaction}.",
               {"account_id": string("Account being checked."), "item_id": string("Requested item."),
                "date": string("Requested date in YYYY-MM-DD form.")}, ["account_id", "item_id", "date"]),
        schema(names["create"], f"Create a {domain.transaction}; call only after a successful eligibility check.",
               {"account_id": string("Eligible account."), "item_id": string("Available item."),
                "date": string("Confirmed date."), "slot": string("Confirmed time slot."),
                "eligibility_token": string("Token returned by the eligibility tool.")},
               ["account_id", "item_id", "date", "slot", "eligibility_token"]),
        schema(names["get_txn"], f"Retrieve the latest status and terms of a {domain.transaction}.",
               {"request_id": string(f"Identifier of the {domain.transaction}.")}, ["request_id"]),
        schema(names["update"], f"Update the date or slot of an existing {domain.transaction}.",
               {"request_id": string("Request to update."), "date": string("New date."),
                "slot": string("New time slot.")}, ["request_id", "date", "slot"]),
        schema(names["cancel"], f"Cancel an existing {domain.transaction} after explicit user confirmation.",
               {"request_id": string("Request to cancel."), "reason": string("Reason given by the user.")},
               ["request_id", "reason"]),
        schema(names["history"], f"List recent {domain.transaction} records for an account.",
               {"account_id": string("Account whose history is requested.")}, ["account_id"]),
        schema(names["notice"], f"Send a notice about an existing {domain.transaction}.",
               {"request_id": string("Request referenced by the notice."),
                "channel": {"type": "string", "enum": ["email", "sms"]}}, ["request_id", "channel"]),
        schema(names["hours"], f"Return public opening hours for the {domain.organization}.",
               {"location": string("Service location.")}, ["location"]),
    ]
    return {"names": names, "tools": tools}


def call(name: str, arguments: dict[str, Any]) -> dict[str, Any]:
    return {
        "role": "assistant",
        "content": "",
        "tool_calls": [{"type": "function", "function": {"name": name, "arguments": arguments}}],
    }


def result(name: str, payload: dict[str, Any]) -> dict[str, Any]:
    return {"role": "tool", "name": name, "content": json.dumps(payload, ensure_ascii=False, sort_keys=True)}


def text(content: str) -> dict[str, str]:
    return {"role": "assistant", "content": content}


def ids(domain: Domain, number: int) -> dict[str, str]:
    prefix = "".join(part[0] for part in domain.slug.split("_"))[:3].upper()
    return {
        "account": f"{prefix}-A{number:05d}",
        "item": f"{prefix}-I{(number * 7) % 10000:04d}",
        "request": f"{prefix}-R{(number * 13) % 100000:05d}",
        "token": hashlib.sha1(f"{domain.slug}:{number}".encode()).hexdigest()[:12],
    }


def system_prompt(domain: Domain) -> str:
    return (
        f"You are the service assistant for a {domain.organization}. Follow policy exactly. "
        "Use tools for account records, availability, eligibility, and changes; never invent record values. "
        f"Before creating a {domain.transaction}, verify the account, find an available {domain.item}, and run the eligibility check. "
        "Cancellation requires explicit user confirmation. Ask for missing required information instead of guessing. "
        "After an operation is complete, answer with a concise summary grounded in tool results and do not call another tool."
    )


def episode(domain: Domain, number: int, variant: int, rng: random.Random) -> dict[str, Any]:
    made = toolset(domain)
    n = made["names"]
    x = ids(domain, number)
    date = f"2026-{1 + number % 12:02d}-{1 + (number * 3) % 27:02d}"
    new_date = f"2026-{1 + (number + 1) % 12:02d}-{1 + (number * 5) % 27:02d}"
    location = rng.choice(["North site", "Riverside site", "Central site", "West site"])
    slot = rng.choice(["09:00", "11:30", "14:00", "16:30"])
    email = f"user{number}@example.org"
    base = [{"role": "system", "content": system_prompt(domain)}]

    if variant == 0:  # full multi-step create flow
        msgs = base + [
            {"role": "user", "content": f"Please arrange a {domain.transaction} for account {x['account']} at {location} on {date}."},
            call(n["get_account"], {"account_id": x["account"]}),
            result(n["get_account"], {"account_id": x["account"], "status": "active", "level": "standard"}),
            call(n["list_items"], {"location": location, "date": date}),
            result(n["list_items"], {"location": location, "date": date,
                                      "available": [{"item_id": x["item"], "slot": slot}]}),
            call(n["check"], {"account_id": x["account"], "item_id": x["item"], "date": date}),
            result(n["check"], {"eligible": True, "eligibility_token": x["token"], "expires": date}),
            call(n["create"], {"account_id": x["account"], "item_id": x["item"], "date": date,
                                "slot": slot, "eligibility_token": x["token"]}),
            result(n["create"], {"request_id": x["request"], "status": "confirmed", "date": date,
                                  "slot": slot, "item_id": x["item"]}),
            text(f"Confirmed {x['request']} for {x['item']} on {date} at {slot}."),
        ]
    elif variant == 1:  # discover account, inspect existing request, then update
        msgs = base + [
            {"role": "user", "content": f"My email is {email}. Move my current {domain.transaction} to {new_date} at {slot}."},
            call(n["find_account"], {"email": email}),
            result(n["find_account"], {"email": email, "account_id": x["account"], "matches": 1}),
            call(n["history"], {"account_id": x["account"]}),
            result(n["history"], {"account_id": x["account"],
                                   "records": [{"request_id": x["request"], "status": "confirmed"}]}),
            call(n["get_txn"], {"request_id": x["request"]}),
            result(n["get_txn"], {"request_id": x["request"], "status": "confirmed", "can_update": True,
                                   "date": date, "slot": "09:00"}),
            call(n["update"], {"request_id": x["request"], "date": new_date, "slot": slot}),
            result(n["update"], {"request_id": x["request"], "status": "confirmed",
                                  "date": new_date, "slot": slot}),
            text(f"Updated {x['request']} to {new_date} at {slot}."),
        ]
    elif variant == 2:  # cancellation confirmation boundary
        msgs = base + [
            {"role": "user", "content": f"Cancel request {x['request']}."},
            text(f"Cancellation is permanent. Please confirm that you want me to cancel {x['request']} and provide a reason."),
            {"role": "user", "content": "Yes, cancel it because my schedule changed."},
            call(n["get_txn"], {"request_id": x["request"]}),
            result(n["get_txn"], {"request_id": x["request"], "status": "confirmed", "can_cancel": True}),
            call(n["cancel"], {"request_id": x["request"], "reason": "schedule changed"}),
            result(n["cancel"], {"request_id": x["request"], "status": "cancelled"}),
            text(f"Request {x['request']} is now cancelled."),
        ]
    elif variant == 3:  # ask for required facts, then use them
        msgs = base + [
            {"role": "user", "content": f"I need a {domain.transaction}."},
            text("What is your account ID, preferred location, and requested date?"),
            {"role": "user", "content": f"Account {x['account']}, {location}, {date}."},
            call(n["get_account"], {"account_id": x["account"]}),
            result(n["get_account"], {"account_id": x["account"], "status": "active"}),
            call(n["list_items"], {"location": location, "date": date}),
            result(n["list_items"], {"location": location, "date": date,
                                      "available": [{"item_id": x["item"], "slot": slot}]}),
            text(f"I found {x['item']} available at {slot}. Which time should I reserve?"),
        ]
    elif variant == 4:  # grounded stop after tool result
        msgs = base + [
            {"role": "user", "content": f"Can account {x['account']} make a new {domain.transaction} today?"},
            call(n["get_account"], {"account_id": x["account"]}),
            result(n["get_account"], {"account_id": x["account"], "status": "suspended",
                                       "reason": "verification expired"}),
            text(f"Account {x['account']} is suspended because its verification expired, so I cannot create a request yet."),
        ]
    elif variant == 5:  # one useful call, then stop
        msgs = base + [
            {"role": "user", "content": f"What are the opening hours at {location}?"},
            call(n["hours"], {"location": location}),
            result(n["hours"], {"location": location, "hours": "09:00-17:00", "timezone": "local"}),
            text(f"The {location} location is open from 09:00 to 17:00 local time."),
        ]
    elif variant == 6:  # current terms must be read before action
        msgs = base + [
            {"role": "user", "content": f"Send an email notice for request {x['request']}."},
            call(n["get_txn"], {"request_id": x["request"]}),
            result(n["get_txn"], {"request_id": x["request"], "status": "confirmed", "notice_allowed": True}),
            call(n["notice"], {"request_id": x["request"], "channel": "email"}),
            result(n["notice"], {"request_id": x["request"], "channel": "email", "delivered": True}),
            text(f"The email notice for {x['request']} was delivered."),
        ]
    else:  # irrelevant request: tools must not be hallucinated
        msgs = base + [
            {"role": "user", "content": "Could you write and publish a newspaper advertisement for me?"},
            text(f"I cannot publish advertisements with the {domain.organization}'s available tools."),
        ]

    return {
        "source": f"policy_sim_v10_{variant}",
        "source_group": f"policy-v10-{domain.slug}-{number}",
        "tools": made["tools"],
        "messages": msgs,
    }


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out-dir", type=Path, default=Path("training_data_policy_v10"))
    parser.add_argument("--episodes-per-domain", type=int, default=240)
    parser.add_argument("--validation-per-domain", type=int, default=16)
    parser.add_argument("--seed", type=int, default=20261101)
    args = parser.parse_args()
    if args.validation_per_domain >= args.episodes_per_domain:
        raise SystemExit("validation-per-domain must be smaller than episodes-per-domain")

    rng = random.Random(args.seed)
    train: list[dict[str, Any]] = []
    validation: list[dict[str, Any]] = []
    by_source: dict[str, int] = {}
    for domain_index, domain in enumerate(DOMAINS):
        order = list(range(args.episodes_per_domain))
        rng.shuffle(order)
        validation_numbers = set(order[: args.validation_per_domain])
        for local_number in range(args.episodes_per_domain):
            number = domain_index * 10000 + local_number
            row = episode(domain, number, local_number % 8, rng)
            by_source[row["source"]] = by_source.get(row["source"], 0) + 1
            (validation if local_number in validation_numbers else train).append(row)
    rng.shuffle(train)
    rng.shuffle(validation)
    write_jsonl(args.out_dir / "train.jsonl", train)
    write_jsonl(args.out_dir / "validation.jsonl", validation)
    manifest = {
        "generator": "code/generate_policy_v10.py",
        "seed": args.seed,
        "policy": "Original deterministic simulator; never reads HW3 PUBLIC or peer code/data/weights.",
        "domains": len(DOMAINS),
        "episodes_per_domain": args.episodes_per_domain,
        "train_episodes": len(train),
        "validation_episodes": len(validation),
        "all_by_source": dict(sorted(by_source.items())),
    }
    (args.out_dir / "manifest.json").write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    print(json.dumps(manifest, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
