#!/usr/bin/env python3
"""Create original policy-branch conversations with observable prerequisites.

Each case offers several plausible checks.  A short policy selects one check
and sometimes requires a field that is optional in the API schema.  Results
are generated from the same case state and no homework test record is read.
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
from generate_policy_v11 import MORE_DOMAINS, validate_case


RULES = ("orientation", "clearance", "deposit", "priority")
USER_TEMPLATES = (
    "Please confirm a {item} at {location} on {date} for account {account} at {slot}.",
    "Account {account} needs a {item} at {location} for {date}, {slot}. Please make the booking now.",
    "Could you arrange and confirm a {item} at {location} on {date} at {slot}? My account is {account}.",
    "I want a confirmed {item} booking, account {account}, {location}, {date}, {slot}.",
)


def extra_tools(domain: Domain) -> dict[str, dict[str, Any]]:
    slug = domain.slug
    return {
        "orientation": schema(f"verify_{slug}_orientation", "Verify this account's current safety orientation before booking.",
                              {"account_id": {"type": "string"}}, ["account_id"]),
        "clearance": schema(f"check_{slug}_clearance", "Check whether this account has the clearance level needed for this service.",
                            {"account_id": {"type": "string"}}, ["account_id"]),
        "deposit": schema(f"quote_{slug}_deposit", "Get the deposit amount and reference for the selected available item.",
                          {"account_id": {"type": "string"}, "item_id": {"type": "string"}},
                          ["account_id", "item_id"]),
        "priority": schema(f"read_{slug}_booking_policy", "Read the current booking priority and confirmation requirements.",
                           {"location": {"type": "string"}}, ["location"]),
    }


def build(domain: Domain, serial: int, rule: str, family: int, wording: int, rng: random.Random) -> dict[str, Any]:
    made = toolset(domain)
    names = made["names"]
    special = extra_tools(domain)
    tools = json.loads(json.dumps(made["tools"])) + list(special.values())
    check_props = tools[4]["function"]["parameters"]["properties"]
    check_props["orientation_code"] = {"type": "string", "description": "Required by policy when an orientation is needed."}
    check_props["clearance_level"] = {"type": "string", "description": "Required by policy when clearance is needed."}
    create_props = tools[5]["function"]["parameters"]["properties"]
    create_props["deposit_reference"] = {"type": "string", "description": "Required by policy for deposit-controlled bookings."}
    create_props["priority"] = {"type": "string", "enum": ["standard", "urgent"],
                                "description": "Required by policy for priority-controlled bookings."}
    create_props["status"] = {"type": "string", "enum": ["draft", "confirmed"],
                              "description": "Use confirmed for an immediate confirmed booking."}
    x = ids(domain, serial)
    location = rng.choice(("Harbor Site", "South Annex", "Hill Center", "North Workshop"))
    date = rng.choice(("2027-01-22", "2027-02-18", "2027-04-09", "2027-08-27"))
    other_date = rng.choice(("2027-09-03", "2027-10-14", "2027-11-21"))
    slot = rng.choice(("09:30", "11:00", "14:30", "16:00"))
    check_value = f"OR-{serial:06d}"
    deposit_ref = f"DEP-{serial:06d}"
    policy_lines = {
        "orientation": ("Before listing availability, verify the account's safety orientation. "
                        "Pass the returned orientation_code to the eligibility check."),
        "clearance": ("Before listing availability, check the account's clearance. "
                      "Pass the returned clearance_level to the eligibility check."),
        "deposit": ("After finding an available item, request its deposit quote. "
                    "Pass the returned deposit_reference when creating the booking."),
        "priority": ("Read the current booking policy for this location before checking the account. "
                     "Pass its priority to the create call and set status=confirmed."),
    }
    system = (f"You are assisting the {domain.organization}. Follow this service policy: "
              "Never invent record values. Check the account before creating a booking. "
              f"{policy_lines[rule]} Check eligibility before creating. "
              "Ask for missing date or account details. Read the current request before an update. "
              "Get explicit confirmation before cancellation. Stop on denial or no availability; "
              "after success, give a short factual answer.")
    messages: list[dict[str, Any]] = [{"role": "system", "content": system}]

    def use(name: str, arguments: dict[str, Any], payload: dict[str, Any]) -> None:
        messages.append(call(name, arguments))
        messages.append(result(name, payload))

    def text(body: str) -> None:
        messages.append({"role": "assistant", "content": body})

    if family in (0, 1, 2, 3, 4):
        user = USER_TEMPLATES[wording].format(item=domain.item, location=location, date=date,
                                              account=x["account"], slot=slot)
        if family == 4:
            messages.append({"role": "user", "content":
                             f"I need a confirmed {domain.item} booking for account {x['account']} at {location}."})
            text("What date and time would you like for the booking?")
            messages.append({"role": "user", "content": f"Use {date} at {slot}."})
        else:
            messages.append({"role": "user", "content": user})
        if rule == "priority":
            use(special[rule]["function"]["name"], {"location": location},
                {"location": location, "priority": "urgent" if serial % 3 == 0 else "standard",
                 "approval": family != 1})
            if family == 1:
                text("The current booking policy does not approve this request.")
        if family != 1 or rule != "priority":
            use(names["get_account"], {"account_id": x["account"]},
                {"account_id": x["account"], "status": "suspended" if family == 2 else "active"})
            if family == 2:
                text("The account is suspended, so I cannot create this booking.")
            else:
                if rule in ("orientation", "clearance"):
                    key = "orientation_code" if rule == "orientation" else "clearance_level"
                    value = check_value if rule == "orientation" else "level-two"
                    use(special[rule]["function"]["name"], {"account_id": x["account"]},
                        {"account_id": x["account"], "approved": family != 1, key: value})
                    if family == 1:
                        text(f"The required {rule} check was not approved, so I cannot make the booking.")
                if family != 1 or rule == "deposit":
                    use(names["list_items"], {"location": location, "date": date},
                        {"location": location, "date": date, "available": [] if family == 3 else
                         [{"item_id": x["item"], "slot": slot}]})
                    if family == 3:
                        text(f"There is no available {domain.item} at {location} on {date}.")
                    else:
                        if rule == "deposit":
                            use(special[rule]["function"]["name"],
                                {"account_id": x["account"], "item_id": x["item"]},
                                {"approved": family != 1, "amount": 35.0,
                                 "deposit_reference": deposit_ref})
                            if family == 1:
                                text("The deposit check was denied, so I cannot make the booking.")
                        if family != 1:
                            check_args = {"account_id": x["account"], "item_id": x["item"], "date": date}
                            if rule == "orientation":
                                check_args["orientation_code"] = check_value
                            if rule == "clearance":
                                check_args["clearance_level"] = "level-two"
                            use(names["check"], check_args,
                                {"eligible": True, "eligibility_token": x["token"],
                                 "account_id": x["account"], "item_id": x["item"]})
                            create_args = {"account_id": x["account"], "item_id": x["item"],
                                           "date": date, "slot": slot, "eligibility_token": x["token"]}
                            if rule == "deposit":
                                create_args["deposit_reference"] = deposit_ref
                            if rule == "priority":
                                create_args.update(priority="urgent" if serial % 3 == 0 else "standard",
                                                   status="confirmed")
                            use(names["create"], create_args,
                                {"request_id": x["request"], "status": "confirmed",
                                 "date": date, "slot": slot})
                            text(f"Request {x['request']} is confirmed for {date} at {slot}.")
    elif family == 5:
        messages.append({"role": "user", "content":
                         f"Move my existing {domain.transaction} {x['request']} to {other_date} at {slot}."})
        use(names["get_txn"], {"request_id": x["request"]},
            {"request_id": x["request"], "status": "confirmed", "can_update": True})
        use(names["update"], {"request_id": x["request"], "date": other_date, "slot": slot},
            {"request_id": x["request"], "status": "confirmed", "date": other_date, "slot": slot})
        text(f"Request {x['request']} has been moved to {other_date} at {slot}.")
    elif family == 6:
        messages.append({"role": "user", "content": f"Please cancel {x['request']}."})
        text(f"Please confirm you want to cancel {x['request']} and give a reason.")
        messages.append({"role": "user", "content": "Yes, cancel it because the event was postponed."})
        use(names["get_txn"], {"request_id": x["request"]},
            {"request_id": x["request"], "status": "confirmed", "can_cancel": True})
        use(names["cancel"], {"request_id": x["request"], "reason": "event was postponed"},
            {"request_id": x["request"], "status": "cancelled"})
        text(f"Request {x['request']} is cancelled.")
    else:
        question, answer = rng.choice((
            ("Can you write a poem about the sea?", "Blue water meets the morning light; waves turn the shore to silver."),
            ("Tell me a joke about computers.", "Why did the computer visit the doctor? It had a virus."),
            ("Can you reserve a table at an unrelated restaurant?",
             "I cannot make restaurant reservations with the available service tools."),
        ))
        messages.append({"role": "user", "content": question})
        text(answer)
    row = {"source": f"policy_branch_v12_{rule}_{family}",
           "source_group": f"policy-branch-v12-{domain.slug}-{serial}",
           "tools": tools, "messages": messages}
    validate_case(row)
    return row


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out-dir", type=Path, default=Path("training_data_policy_branch_v12"))
    parser.add_argument("--per-domain", type=int, default=128)
    parser.add_argument("--validation-per-domain", type=int, default=8)
    parser.add_argument("--seed", type=int, default=20261029)
    args = parser.parse_args()
    if args.per_domain % 32 or args.validation_per_domain >= args.per_domain:
        raise SystemExit("per-domain must be a multiple of 32, with a smaller validation count")
    rng = random.Random(args.seed)
    train, validation = [], []
    for index, domain in enumerate(DOMAINS + MORE_DOMAINS):
        order = list(range(args.per_domain))
        rng.shuffle(order)
        held_out = set(order[:args.validation_per_domain])
        for local in range(args.per_domain):
            serial = 900000 + index * 1000 + local
            rule = RULES[(local // 8) % 4]
            family = local % 8
            row = build(domain, serial, rule, family, (local // 32) % 4, rng)
            (validation if local in held_out else train).append(row)
    rng.shuffle(train)
    rng.shuffle(validation)
    args.out_dir.mkdir(parents=True, exist_ok=True)
    for name, rows in (("train", train), ("validation", validation)):
        with (args.out_dir / f"{name}.jsonl").open("w", encoding="utf-8") as stream:
            for row in rows:
                stream.write(json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n")
    manifest = {"generator": "code/generate_policy_branch_v12.py", "seed": args.seed,
                "policy_configurations": 4 * len(DOMAINS + MORE_DOMAINS),
                "service_domains": len(DOMAINS + MORE_DOMAINS),
                "train_conversations": len(train), "validation_conversations": len(validation),
                "train_by_source": dict(collections.Counter(row["source"] for row in train)),
                "public_test_usage": "none",
                "code_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest()}
    (args.out_dir / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({k: manifest[k] for k in ("service_domains", "policy_configurations",
                                                "train_conversations", "validation_conversations")}, indent=2))


if __name__ == "__main__":
    main()
