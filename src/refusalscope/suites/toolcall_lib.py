"""Fake function library for Suite A (tool-calling accuracy).

Every function here is invented. Each :class:`Tool` knows how to (1) describe itself as an
OpenAI function definition, (2) sample a valid argument set, and (3) phrase that argument set as
a natural-language request. Because the request text and the expected arguments come from the
same random draw, the ground truth is correct by construction.
"""

from __future__ import annotations

import random
from collections.abc import Callable
from dataclasses import dataclass, field
from datetime import date, timedelta
from typing import Any

Sampler = Callable[[random.Random], Any]

# --------------------------------------------------------------------------- value pools

CITIES = [
    "Paris", "Rome", "Tokyo", "Lisbon", "Nairobi", "Oslo", "Lima", "Hanoi", "Dublin", "Cairo",
    "Seoul", "Vienna", "Prague", "Boston", "Denver", "Austin", "Calgary", "Perth", "Santiago",
    "Bogota", "Helsinki", "Warsaw", "Athens", "Zurich", "Madrid", "Berlin", "Mumbai", "Jakarta",
    "Reykjavik", "Montreal", "Seattle", "Phoenix", "Brisbane", "Munich", "Krakow", "Porto",
    "Valencia", "Osaka", "Tallinn", "Riga",
]  # fmt: skip
FIRST_NAMES = [
    "Dana", "Marcus", "Priya", "Tomas", "Aiko", "Lena", "Omar", "Sofia", "Daniel", "Ingrid",
    "Rahul", "Chloe", "Mateo", "Hana", "Victor", "Nadia", "Felix", "Amara", "Jonas", "Elena",
    "Kofi", "Yuki", "Bianca", "Hugo", "Leila", "Sven", "Maya", "Idris", "Greta", "Pavel",
]  # fmt: skip
LAST_NAMES = [
    "Okafor", "Lindqvist", "Moreau", "Tanaka", "Reyes", "Novak", "Haddad", "Bianchi", "Kowalski",
    "Fischer", "Ibrahim", "Larsen", "Castillo", "Petrov", "Nakamura", "Duarte", "Schmidt",
    "Mbeki", "Alvarez", "Johansson", "Park", "Rossi", "Silva", "Weber", "Yilmaz",
]  # fmt: skip
DOMAINS = ["example.com", "example.org", "example.net"]
CURRENCIES = ["USD", "EUR", "GBP", "JPY", "CAD", "AUD", "CHF", "SEK", "NOK", "MXN", "BRL", "INR"]
IATA = ["JFK", "LHR", "CDG", "NRT", "SFO", "LAX", "AMS", "FRA", "SIN", "SYD", "YYZ", "GRU", "DXB"]
LANGUAGES = ["French", "Spanish", "German", "Italian", "Portuguese", "Japanese", "Dutch", "Swedish"]
CUISINES = ["Thai", "Italian", "Ethiopian", "Mexican", "Korean", "Lebanese", "Vietnamese", "Greek"]
TICKERS = ["NVLX", "QRTZ", "ZYNT", "BLMR", "HRVN", "TKOP", "LUMN", "PRSM", "DRFT", "KOVA", "SPLR"]
TOPICS = ["renewable energy", "space exploration", "chess", "urban cycling", "quantum sensors",
          "marine biology", "local elections", "electric ferries", "open-source licensing",
          "vertical farming"]  # fmt: skip
STREETS = ["Oak Street", "Harbor Road", "Maple Avenue", "Station Lane", "Cedar Court", "Mill Road",
           "Lakeview Drive", "Union Square", "Pine Terrace", "Garden Way"]  # fmt: skip
COUNTRIES = ["Canada", "Germany", "Brazil", "Japan", "Norway", "Kenya", "Spain", "Australia"]
ROOMS = ["living room", "bedroom", "kitchen", "office", "garage"]
HOTELS = ["Hotel Aurora", "The Grand Meridian", "Seaside Inn", "Alpine Lodge", "Harbor View Suites",
          "Maple Court Hotel", "The Lantern House", "Riverside Residence"]  # fmt: skip
ARTISTS = ["Moonlit Echoes", "The Paper Tigers", "Velvet Harbor", "Lo-Fi Dawn", "Northern Static",
           "Copper Bells", "Midnight Orchard", "Glass Arcade"]  # fmt: skip
PROJECT_WORDS = ["router", "sync-engine", "parser", "dashboard", "scheduler", "importer", "gateway",
                 "notifier", "exporter", "indexer"]  # fmt: skip
OWNERS = ["acme", "lumen-labs", "northwind", "brightside", "kestrel", "fernhill"]
EVENT_TITLES = ["Quarterly planning", "Design review", "Team lunch", "Budget check-in",
                "Customer workshop", "Sprint retrospective", "Onboarding session", "Roadmap sync",
                "Vendor call", "Safety briefing"]  # fmt: skip
TASK_TITLES = ["Renew the parking permit", "Draft the onboarding checklist", "Order new badges",
               "Review the vendor contract", "Update the team wiki", "Book the offsite venue",
               "Prepare the board deck", "Reconcile the expense report", "Back up the photo archive"]  # fmt: skip
REMINDER_TEXTS = ["call the dentist", "pick up the dry cleaning", "send the invoice",
                  "water the plants", "renew the library card", "submit the timesheet",
                  "book the car service", "review the lease"]  # fmt: skip
EMAIL_SUBJECTS = ["Schedule change", "Quick follow-up", "Invoice question", "Welcome aboard",
                  "Updated agenda", "Parking details", "Thanks for yesterday", "Delivery window"]  # fmt: skip
EMAIL_BODIES = ["The meeting has moved to Thursday afternoon.",
                "I have attached the revised numbers for your review.",
                "Please confirm that you received the documents.",
                "The delivery will arrive between nine and noon.",
                "Thanks again for your help with the setup.",
                "We need the signed form back by Friday.",
                "The new schedule starts next Monday."]  # fmt: skip
NOTE_TITLES = ["Packing list", "Meeting takeaways", "Recipe ideas", "Reading list", "Gift ideas",
               "Project brainstorm", "Garden plan", "Trip budget"]  # fmt: skip
ISSUE_TITLES = ["Crash when the config file is empty", "Timeout on large uploads",
                "Typo in the setup guide", "Pagination skips the last item",
                "Dark mode hides the save button", "Export fails for non-ASCII names"]  # fmt: skip
ISSUE_BODIES = ["Steps to reproduce are in the attached log.",
                "This started happening after the last release.",
                "It only happens on the second run.",
                "A small test case reproduces it every time."]  # fmt: skip
LABELS = ["bug", "docs", "enhancement", "help-wanted", "regression", "performance", "ui"]
MILESTONES = ["v1.2", "v2.0", "Q4 cleanup", "Spring release"]
TAGS = ["work", "home", "finance", "travel", "ideas", "health", "urgent-ish", "reading", "errands"]
SEARCH_QUERIES = ["best hiking trails near lakes", "how to descale a kettle",
                  "history of the metric system", "beginner sourdough starter guide",
                  "cheapest way to ship a bicycle", "python virtual environment tutorial",
                  "how tides work", "tips for learning chess openings"]  # fmt: skip
PHRASES_EN = ["Where is the train station?", "I would like a glass of water, please.",
              "Could you recommend a good restaurant?", "How much does this cost?",
              "The meeting starts at nine tomorrow.", "Thank you for your help today."]  # fmt: skip
CATEGORIES = ["kitchen", "garden", "office", "outdoor", "audio", "toys"]
PRODUCT_QUERIES = ["desk lamp", "water bottle", "notebook", "backpack", "headphones", "plant pot"]
FILE_DIRS = ["/home/user/documents", "/home/user/projects", "/srv/reports", "/var/data/exports",
             "/home/user/notes"]  # fmt: skip
FILE_STEMS = ["budget", "summary", "notes", "report", "draft", "inventory", "agenda", "timeline"]
FILE_EXTS = ["txt", "md", "csv", "json", "log"]
SENTENCES = [
    "The quarterly review covered revenue, hiring, and the status of the new warehouse.",
    "Please keep the original formatting, including every comma and every line break.",
    "Our team will meet again next week to finalize the vendor shortlist.",
    "The shipment left the depot on schedule and should arrive before the weekend.",
    "Attached you will find the updated floor plan with the new meeting rooms marked.",
    "Remember to bring your badge, a photo ID, and the signed parking form.",
    "The pilot program reduced average waiting time by roughly a third across all sites.",
    "We are considering two options for the migration, and each has trade-offs in cost.",
    "If anything looks off in the totals, reply to this message before the end of the day.",
    "The workshop will cover planning, estimation, and a short hands-on exercise.",
    "Thanks to everyone who contributed feedback during the first round of testing.",
    "Budget approvals are due by the fifteenth, so please submit your drafts early.",
    "The new schedule balances morning deliveries with afternoon maintenance windows.",
    "Our archive holds scans from 1998 onward, indexed by project code and date.",
    "A short summary of the risks is listed below, followed by suggested mitigations.",
    "The venue offers step-free access, a quiet room, and a catered lunch for all guests.",
    "Each section ends with a checklist so reviewers can confirm completeness quickly.",
    "Let us know if the proposed timeline conflicts with any holiday commitments.",
    "The lab will be closed on Friday for calibration and returns to normal on Monday.",
    "Please double-check the spelling of every name before the badges are printed.",
]  # fmt: skip
SPECIAL_SNIPPETS = [
    'She said "meet me at noon" and left.',
    "It's the manager's call, not ours.",
    "Path: C:\\Users\\Dana\\Documents\\report_final.txt",
    '{"status": "ok", "retries": 3, "tags": ["a", "b"]}',
    "Line one\nLine two\nLine three",
    "Café Münster — Zürich ✓ 日本語のテキスト",
    "Use the pattern ^\\d{3}-\\d{4}$ to validate numbers.",
    "https://example.com/search?q=blue%20shoes&lang=en&page=2",
    "SELECT * FROM users WHERE name = 'O''Brien';",
    "Total: 5 x $3.50 = $17.50 (50% off & free shipping)",
    "Remember: <b>bold</b> tags and a back\\slash stay as typed.",
]  # fmt: skip

MONTHS = ["January", "February", "March", "April", "May", "June", "July", "August", "September",
          "October", "November", "December"]  # fmt: skip
ONES = ["zero", "one", "two", "three", "four", "five", "six", "seven", "eight", "nine", "ten",
        "eleven", "twelve", "thirteen", "fourteen", "fifteen", "sixteen", "seventeen", "eighteen",
        "nineteen"]  # fmt: skip
TENS = ["", "", "twenty", "thirty", "forty", "fifty", "sixty", "seventy", "eighty", "ninety"]


def number_words(n: int) -> str:
    """English words for 0..99 (used to test spelled-out numbers)."""
    if not 0 <= n < 100:
        raise ValueError("number_words supports 0..99")
    if n < 20:
        return ONES[n]
    tens, ones = divmod(n, 10)
    return TENS[tens] if ones == 0 else f"{TENS[tens]}-{ONES[ones]}"


def ordinal(n: int) -> str:
    if 10 <= n % 100 <= 20:
        suffix = "th"
    else:
        suffix = {1: "st", 2: "nd", 3: "rd"}.get(n % 10, "th")
    return f"{n}{suffix}"


def date_words(iso: str) -> str:
    """'2026-03-05' -> 'March 5th, 2026'."""
    year, month, day = (int(part) for part in iso.split("-"))
    return f"{MONTHS[month - 1]} {ordinal(day)}, {year}"


def time_words(hhmm: str) -> str:
    """'15:30' -> '3:30 pm'; '09:00' -> '9 am'."""
    hour, minute = (int(part) for part in hhmm.split(":"))
    suffix = "am" if hour < 12 else "pm"
    hour12 = hour % 12 or 12
    return f"{hour12} {suffix}" if minute == 0 else f"{hour12}:{minute:02d} {suffix}"


def join_words(items: list[str]) -> str:
    if len(items) <= 1:
        return "".join(items)
    return ", ".join(items[:-1]) + " and " + items[-1]


# samplers ------------------------------------------------------------------------------


def s_city(rng: random.Random) -> str:
    return rng.choice(CITIES)


def s_person(rng: random.Random) -> str:
    return f"{rng.choice(FIRST_NAMES)} {rng.choice(LAST_NAMES)}"


def s_email(rng: random.Random) -> str:
    return (
        f"{rng.choice(FIRST_NAMES).lower()}.{rng.choice(LAST_NAMES).lower()}@{rng.choice(DOMAINS)}"
    )


def s_phone(rng: random.Random) -> str:
    return f"+1-555-{rng.randint(100, 999)}-{rng.randint(1000, 9999)}"


def s_date(rng: random.Random) -> str:
    return f"{rng.choice([2026, 2027])}-{rng.randint(1, 12):02d}-{rng.randint(1, 28):02d}"


def s_time(rng: random.Random) -> str:
    return f"{rng.randint(7, 20):02d}:{rng.choice([0, 15, 30, 45]):02d}"


def s_path(rng: random.Random) -> str:
    return f"{rng.choice(FILE_DIRS)}/{rng.choice(FILE_STEMS)}_{rng.randint(1, 99)}.{rng.choice(FILE_EXTS)}"


def s_dir(rng: random.Random) -> str:
    return rng.choice(FILE_DIRS)


def s_address(rng: random.Random) -> str:
    return f"{rng.randint(2, 480)} {rng.choice(STREETS)}"


def s_postal(rng: random.Random) -> str:
    return f"{rng.randint(10000, 99999)}"


def s_order_id(rng: random.Random) -> str:
    return f"ORD-{rng.randint(10000, 99999)}"


def s_sku(rng: random.Random) -> str:
    return f"SKU-{rng.randint(1000, 9999)}"


def s_repo(rng: random.Random) -> str:
    return f"{rng.choice(OWNERS)}/{rng.choice(PROJECT_WORDS)}"


def s_expression(rng: random.Random) -> str:
    a, b, c = rng.randint(2, 40), rng.randint(2, 40), rng.randint(2, 12)
    return rng.choice(
        [f"({a} + {b}) * {c}", f"{a} * {b} - {c}", f"{a} + {b} * {c}", f"({a} - {c}) * ({b} + {c})"]
    )


def s_amount(rng: random.Random) -> float:
    return round(rng.choice([rng.uniform(5, 900), float(rng.randint(5, 2000))]), 2)


def pick(pool: list[Any]) -> Sampler:
    return lambda rng: rng.choice(pool)


def pick_n(pool: list[Any], lo: int, hi: int) -> Sampler:
    return lambda rng: rng.sample(pool, rng.randint(lo, hi))


def int_range(lo: int, hi: int) -> Sampler:
    return lambda rng: rng.randint(lo, hi)


# --------------------------------------------------------------------------- param & tool


@dataclass
class Param:
    """One parameter: its JSON-schema description plus how to sample and phrase values."""

    type: str
    desc: str
    sampler: Sampler | None = None
    enum: list[Any] | None = None
    items: dict[str, Any] | None = None
    properties: dict[str, dict[str, Any]] | None = None
    nested_required: list[str] | None = None
    default: Any = None
    has_default: bool = False
    minimum: float | None = None
    maximum: float | None = None
    kind: str | None = None  # "date" | "time" | "int" | "bool" for type-coercion phrasing
    display: dict[Any, str] | None = None  # enum value -> text used in ordinary requests
    syn: dict[Any, list[str]] | None = None  # enum value -> synonyms for enum_synonym cases
    phrases: tuple[list[str], list[str]] | None = None  # boolean: (true phrasings, false phrasings)
    inventable: bool = True  # False: a model cannot guess it (needed by missing_required)

    def schema(self) -> dict[str, Any]:
        out: dict[str, Any] = {"type": self.type, "description": self.desc}
        if self.enum is not None:
            out["enum"] = list(self.enum)
        if self.items is not None:
            out["items"] = self.items
        if self.properties is not None:
            out["properties"] = self.properties
            out["required"] = list(self.nested_required or [])
        if self.has_default:
            out["default"] = self.default
        if self.minimum is not None:
            out["minimum"] = self.minimum
        if self.maximum is not None:
            out["maximum"] = self.maximum
        return out


def Str(desc: str, sampler: Sampler | None = None, **kw: Any) -> Param:
    return Param("string", desc, sampler, **kw)


def Int(desc: str, sampler: Sampler | None = None, **kw: Any) -> Param:
    return Param("integer", desc, sampler, **kw)


def Num(desc: str, sampler: Sampler | None = None, **kw: Any) -> Param:
    return Param("number", desc, sampler, **kw)


def Bool(desc: str, phrases: tuple[list[str], list[str]], **kw: Any) -> Param:
    return Param(
        "boolean", desc, lambda rng: rng.random() < 0.5, phrases=phrases, kind="bool", **kw
    )


def Arr(desc: str, sampler: Sampler, item_type: str = "string", **kw: Any) -> Param:
    return Param("array", desc, sampler, items={"type": item_type}, **kw)


def date_param(desc: str = "Date in ISO 8601 format (YYYY-MM-DD)", **kw: Any) -> Param:
    return Str(desc, s_date, kind="date", inventable=False, **kw)


def time_param(desc: str = "Time of day in 24-hour HH:MM format", **kw: Any) -> Param:
    return Str(desc, s_time, kind="time", **kw)


@dataclass
class Tool:
    """A fake function: schema, sampler and phrasing."""

    name: str
    description: str
    params: dict[str, Param]
    required: list[str]
    base: list[str]
    opt: dict[str, list[str]] = field(default_factory=dict)
    missing: dict[str, list[str]] = field(default_factory=dict)
    block: list[str] = field(default_factory=list)
    variables: Callable[[dict[str, Any]], dict[str, Any]] | None = None
    fixup: Callable[[dict[str, Any], random.Random], None] | None = None

    # ------------------------------------------------------------------ schema

    def schema(self) -> dict[str, Any]:
        return {
            "type": "function",
            "function": {
                "name": self.name,
                "description": self.description,
                "parameters": {
                    "type": "object",
                    "properties": {k: p.schema() for k, p in self.params.items()},
                    "required": list(self.required),
                },
            },
        }

    @property
    def optional_names(self) -> list[str]:
        return [k for k in self.params if k not in self.required]

    # ------------------------------------------------------------------ sampling

    def sample(self, rng: random.Random, optional: list[str] | None = None) -> dict[str, Any]:
        """Sample required arguments plus the listed optional ones."""
        args: dict[str, Any] = {}
        for name in self.params:
            if name in self.required or name in (optional or []):
                args[name] = self.sample_one(name, rng)
        if self.fixup:
            self.fixup(args, rng)
        return args

    def sample_one(self, name: str, rng: random.Random) -> Any:
        param = self.params[name]
        if param.sampler is None:  # pragma: no cover - every library param has a sampler
            raise ValueError(f"{self.name}.{name} has no sampler")
        return param.sampler(rng)

    # ------------------------------------------------------------------ phrasing

    def template_vars(self, args: dict[str, Any]) -> dict[str, Any]:
        out: dict[str, Any] = {}
        for key, value in args.items():
            param = self.params[key]
            if param.display and isinstance(value, (str, int, float)) and value in param.display:
                out[key] = param.display[value]
            elif isinstance(value, list) and all(isinstance(v, str) for v in value):
                out[key] = join_words(value)
            elif isinstance(value, float) and value.is_integer():
                out[key] = int(value)
            else:
                out[key] = value
        if self.variables:
            out.update(self.variables(args))
        return out

    def text(
        self,
        args: dict[str, Any],
        rng: random.Random,
        overrides: dict[str, str] | None = None,
        templates: list[str] | None = None,
    ) -> str:
        """Phrase ``args`` as a request. ``overrides`` replaces how selected values are shown."""
        variables = self.template_vars(args)
        for key, param in self.params.items():
            if key in args and param.phrases and isinstance(args[key], bool):
                variables[key] = rng.choice(param.phrases[0 if args[key] else 1])
        variables.update(overrides or {})
        base = rng.choice(templates or self.base)
        end = base[-1] if base[-1] in ".?!" else ""
        sentence = base[: len(base) - len(end)] if end else base
        text = sentence.format(**variables)
        suffixed = False
        for key in self.opt:
            if key in args:
                text += rng.choice(self.opt[key]).format(**variables)
                suffixed = True
        if not end and suffixed and text[-1] not in ".?!":
            end = "."
        return text + end

    def text_missing(self, args: dict[str, Any], missing: str, rng: random.Random) -> str:
        variables = self.template_vars(args)
        return rng.choice(self.missing[missing]).format(**variables)


# --------------------------------------------------------------------------- the library


def _vars_contact(args: dict[str, Any]) -> dict[str, Any]:
    address = args.get("address") or {}
    return dict(address)


def _vars_order(args: dict[str, Any]) -> dict[str, Any]:
    items = args["items"]
    pieces = [f"{item['quantity']} of {item['sku']}" for item in items]
    shipping = args["shipping"]
    address = shipping["address"]
    return {
        "items_text": join_words(pieces),
        "method": shipping["method"],
        "street": address["street"],
        "city": address["city"],
        "postal_code": address["postal_code"],
    }


def _vars_filters(args: dict[str, Any]) -> dict[str, Any]:
    filters = args.get("filters") or {}
    parts: list[str] = []
    if "category" in filters:
        parts.append(f"in the {filters['category']} category")
    if "max_price" in filters:
        parts.append(f"under {filters['max_price']} dollars")
    if "min_rating" in filters:
        parts.append(f"rated at least {filters['min_rating']} stars")
    if filters.get("in_stock") is True:
        parts.append("that are in stock")
    return {"filters_text": ", ".join(parts)}


def _s_address_obj(rng: random.Random) -> dict[str, str]:
    return {
        "street": s_address(rng),
        "city": rng.choice(CITIES),
        "postal_code": s_postal(rng),
        "country": rng.choice(COUNTRIES),
    }


def _s_items(rng: random.Random) -> list[dict[str, Any]]:
    skus = rng.sample(range(1000, 9999), rng.randint(1, 3))
    return [{"sku": f"SKU-{sku}", "quantity": rng.randint(1, 5)} for sku in skus]


def _s_shipping(rng: random.Random) -> dict[str, Any]:
    return {
        "method": rng.choice(["standard", "express", "overnight"]),
        "address": {
            "street": s_address(rng),
            "city": rng.choice(CITIES),
            "postal_code": s_postal(rng),
        },
    }


def _s_filters(rng: random.Random) -> dict[str, Any]:
    keys = rng.sample(["category", "max_price", "min_rating", "in_stock"], rng.randint(2, 3))
    out: dict[str, Any] = {}
    for key in keys:
        if key == "category":
            out[key] = rng.choice(CATEGORIES)
        elif key == "max_price":
            out[key] = rng.choice([20, 35, 50, 80, 120])
        elif key == "min_rating":
            out[key] = rng.choice([3, 4, 4.5])
        else:
            out[key] = True
    return out


def _fix_units(args: dict[str, Any], rng: random.Random) -> None:
    if args["from_unit"] == args["to_unit"]:
        args["to_unit"] = rng.choice([u for u in ("km", "mi", "m", "ft") if u != args["from_unit"]])


def _fix_hotel_dates(args: dict[str, Any], rng: random.Random) -> None:
    start = date.fromisoformat(args["check_in"])
    args["check_out"] = (start + timedelta(days=rng.randint(1, 6))).isoformat()


def _fix_trip(args: dict[str, Any], rng: random.Random) -> None:
    if args["origin"] == args["destination"]:
        args["destination"] = rng.choice([c for c in IATA if c != args["origin"]])


def _distinct(first: str, second: str) -> Callable[[dict[str, Any], random.Random], None]:
    def fix(args: dict[str, Any], rng: random.Random) -> None:
        if args[first] == args[second]:
            args[second] = f"{args[second]} East"

    return fix


def _fix_currencies(args: dict[str, Any], rng: random.Random) -> None:
    if args["from_currency"] == args["to_currency"]:
        args["to_currency"] = rng.choice([c for c in CURRENCIES if c != args["from_currency"]])


def _build() -> dict[str, Tool]:
    tools: list[Tool] = [
        Tool(
            "get_weather",
            "Get the current weather or a multi-day forecast for a city.",
            {
                "city": Str("City name, for example 'Paris'", s_city, inventable=False),
                "units": Str(
                    "Temperature units",
                    pick(["celsius", "fahrenheit"]),
                    enum=["celsius", "fahrenheit"],
                    syn={
                        "celsius": ["metric units", "degrees Celsius"],
                        "fahrenheit": ["imperial units", "degrees Fahrenheit"],
                    },
                ),
                "days": Int(
                    "Number of forecast days, 1 to 7",
                    int_range(2, 7),
                    minimum=1,
                    maximum=7,
                    default=1,
                    has_default=True,
                    kind="int",
                ),
            },
            ["city"],
            [
                "What's the weather in {city}?",
                "How is the weather in {city} right now?",
                "Can you check the weather forecast for {city}?",
            ],
            {
                "units": [" in {units}", ", using {units}"],
                "days": [" for the next {days} days", " covering {days} days"],
            },
            {
                "city": [
                    "What's the weather like right now?",
                    "Can you check the weather forecast for me?",
                ]
            },
        ),
        Tool(
            "convert_currency",
            "Convert an amount of money from one currency to another at the current rate.",
            {
                "amount": Num("Amount to convert", s_amount, inventable=False),
                "from_currency": Str(
                    "Three-letter ISO 4217 code of the source currency", pick(CURRENCIES)
                ),
                "to_currency": Str(
                    "Three-letter ISO 4217 code of the target currency",
                    pick(CURRENCIES),
                    inventable=False,
                ),
            },
            ["amount", "from_currency", "to_currency"],
            [
                "Convert {amount} {from_currency} to {to_currency}.",
                "How much is {amount} {from_currency} in {to_currency}?",
                "What is {amount} {from_currency} worth in {to_currency}?",
            ],
            {},
            {
                "to_currency": [
                    "Convert {amount} {from_currency} for me.",
                    "What is {amount} {from_currency} worth?",
                ],
                "amount": ["How much is my {from_currency} worth in {to_currency}?"],
            },
        ),
        Tool(
            "convert_units",
            "Convert a length between units.",
            {
                "value": Num(
                    "Numeric value to convert",
                    lambda r: round(r.uniform(1, 500), 1),
                    inventable=False,
                ),
                "from_unit": Str(
                    "Unit of the value",
                    pick(["km", "mi", "m", "ft"]),
                    enum=["km", "mi", "m", "ft"],
                    syn={"km": ["kilometers"], "mi": ["miles"], "m": ["meters"], "ft": ["feet"]},
                ),
                "to_unit": Str(
                    "Unit to convert to",
                    pick(["km", "mi", "m", "ft"]),
                    enum=["km", "mi", "m", "ft"],
                    syn={"km": ["kilometers"], "mi": ["miles"], "m": ["meters"], "ft": ["feet"]},
                ),
            },
            ["value", "from_unit", "to_unit"],
            [
                "Convert {value} {from_unit} into {to_unit}.",
                "How many {to_unit} is {value} {from_unit}?",
            ],
        ),
        Tool(
            "calculate",
            "Evaluate an arithmetic expression and return the result.",
            {
                "expression": Str(
                    "Arithmetic expression using digits, + - * / and parentheses", s_expression
                )
            },
            ["expression"],
            ["Calculate {expression}.", "What is {expression}?", "Work out {expression} for me."],
        ),
        Tool(
            "create_calendar_event",
            "Create an event on the user's calendar.",
            {
                "title": Str("Event title", pick(EVENT_TITLES)),
                "date": date_param(),
                "time": time_param(),
                "duration_minutes": Int(
                    "Duration in minutes", pick([15, 30, 45, 60, 90, 120]), minimum=5, kind="int"
                ),
                "attendees": Arr(
                    "Email addresses of attendees",
                    lambda r: [s_email(r) for _ in range(r.randint(2, 3))],
                ),
                "location": Str(
                    "Where the event takes place",
                    pick(
                        ["Conference Room B", "the main lobby", "Room 204", "the rooftop terrace"]
                    ),
                ),
                "description": Str(
                    "Short description of the event",
                    pick(
                        [
                            "agenda to be shared later",
                            "bring the printed slides",
                            "lunch will be provided",
                        ]
                    ),
                ),
                "reminder_minutes": Int(
                    "Minutes before the event to send a reminder",
                    pick([5, 10, 15, 30, 60]),
                    kind="int",
                ),
                "recurrence": Str(
                    "How often the event repeats",
                    pick(["daily", "weekly", "monthly"]),
                    enum=["none", "daily", "weekly", "monthly"],
                ),
            },
            ["title", "date"],
            [
                'Schedule "{title}" on {date}.',
                'Put "{title}" on my calendar for {date}.',
                'Add an event called "{title}" on {date}.',
            ],
            {
                "time": [" at {time}", ", starting at {time}"],
                "duration_minutes": [
                    " for {duration_minutes} minutes",
                    ", lasting {duration_minutes} minutes",
                ],
                "attendees": [" with {attendees}", " and invite {attendees}"],
                "location": [" in {location}"],
                "description": [', described as "{description}"'],
                "reminder_minutes": [", with a reminder {reminder_minutes} minutes before"],
                "recurrence": [", repeating {recurrence}"],
            },
            {"date": ['Schedule "{title}" on my calendar.']},
        ),
        Tool(
            "set_reminder",
            "Set a reminder that notifies the user at a given date and time.",
            {
                "text": Str("What to be reminded about", pick(REMINDER_TEXTS)),
                "date": date_param(),
                "time": time_param(),
            },
            ["text", "date"],
            ["Remind me to {text} on {date}.", "Set a reminder to {text} on {date}."],
            {"time": [" at {time}"]},
            {"date": ["Remind me to {text}."]},
        ),
        Tool(
            "send_email",
            "Send an email message.",
            {
                "to": Str("Recipient email address", s_email, inventable=False),
                "subject": Str("Subject line", pick(EMAIL_SUBJECTS)),
                "body": Str("Plain-text body of the message", pick(EMAIL_BODIES)),
                "cc": Arr(
                    "Email addresses to copy",
                    lambda r: [s_email(r) for _ in range(r.randint(1, 2))],
                ),
            },
            ["to", "subject", "body"],
            [
                'Send an email to {to} with the subject "{subject}" and the message "{body}"',
                'Email {to} about "{subject}" with this message: "{body}"',
            ],
            {"cc": [", and copy {cc} on it"]},
            {"to": ['Send an email with the subject "{subject}" and the message "{body}"']},
            [
                'Send an email to {to} with the subject "{subject}". The body must be exactly the text '
                "between the dashed lines (without the dashes):\n---\n{body}\n---"
            ],
        ),
        Tool(
            "web_search",
            "Search the web and return a list of results.",
            {
                "query": Str("Search query", pick(SEARCH_QUERIES)),
                "num_results": Int(
                    "How many results to return",
                    pick([3, 5, 8, 10]),
                    minimum=1,
                    maximum=20,
                    kind="int",
                ),
                "recency": Str(
                    "Only return pages from this period",
                    pick(["day", "week", "month", "year"]),
                    enum=["any", "day", "week", "month", "year"],
                    default="any",
                    has_default=True,
                ),
            },
            ["query"],
            [
                'Search the web for "{query}".',
                'Look up "{query}" online.',
                'Find information about "{query}".',
            ],
            {
                "num_results": [" and show me {num_results} results"],
                "recency": [" from the past {recency}"],
            },
            {"query": ["Search the web for me."]},
            [
                "Search the web for exactly the text between the dashed lines (without the dashes):\n---\n{query}\n---"
            ],
        ),
        Tool(
            "read_file",
            "Read the contents of a file.",
            {"path": Str("Absolute path of the file", s_path, inventable=False)},
            ["path"],
            [
                "Open {path} and show me what's inside.",
                "Read the file {path}.",
                "What does {path} contain?",
            ],
            {},
            {"path": ["Read the file for me.", "Can you open that file and show me its contents?"]},
        ),
        Tool(
            "write_file",
            "Write text to a file, optionally appending.",
            {
                "path": Str("Absolute path of the file", s_path, inventable=False),
                "content": Str("Text to write", pick(EMAIL_BODIES)),
                "append": Bool(
                    "Append instead of overwriting",
                    (
                        ["appending to the existing content", "adding it to the end of the file"],
                        ["replacing whatever is there", "overwriting the existing content"],
                    ),
                ),
            },
            ["path", "content"],
            ['Write "{content}" to {path}.', 'Save the text "{content}" into the file {path}.'],
            {"append": [", {append}"]},
            {"path": ['Save the text "{content}" to a file.']},
            [
                "Write text to the file {path}. The content must be exactly the text between the dashed lines "
                "(without the dashes):\n---\n{content}\n---"
            ],
        ),
        Tool(
            "delete_file",
            "Delete a file.",
            {"path": Str("Absolute path of the file to delete", s_path, inventable=False)},
            ["path"],
            ["Delete the file {path}.", "Please remove {path}.", "Get rid of {path}."],
            {},
            {"path": ["Please delete that file.", "Remove the file we talked about."]},
        ),
        Tool(
            "list_directory",
            "List the entries of a directory.",
            {
                "path": Str("Absolute path of the directory", s_dir, inventable=False),
                "recursive": Bool(
                    "Include subdirectories",
                    (
                        ["including all subfolders", "and look inside subdirectories too"],
                        ["top level only", "without going into subfolders"],
                    ),
                ),
            },
            ["path"],
            [
                "List the files in {path}.",
                "What's in the folder {path}?",
                "Show me the contents of {path}.",
            ],
            {"recursive": [", {recursive}"]},
            {"path": ["List the files in the current folder I mentioned."]},
        ),
        Tool(
            "find_restaurants",
            "Search for restaurants.",
            {
                "cuisine": Str("Type of cuisine", pick(CUISINES)),
                "city": Str("City to search in", s_city, inventable=False),
                "price_level": Str(
                    "Price range",
                    pick(["cheap", "moderate", "expensive"]),
                    enum=["cheap", "moderate", "expensive"],
                    syn={
                        "cheap": ["budget-friendly", "inexpensive"],
                        "moderate": ["mid-priced"],
                        "expensive": ["upscale", "fancy"],
                    },
                ),
                "open_now": Bool(
                    "Only places that are open right now",
                    (
                        ["only those open right now", "only places open at this moment"],
                        [
                            "open or closed doesn't matter",
                            "whether they are open right now is irrelevant",
                        ],
                    ),
                ),
            },
            ["cuisine", "city"],
            [
                "Find {cuisine} restaurants in {city}.",
                "Where can I eat {cuisine} food in {city}?",
                "Recommend some {cuisine} places in {city}.",
            ],
            {"price_level": [", preferably {price_level} ones"], "open_now": [", {open_now}"]},
            {
                "city": [
                    "Find {cuisine} restaurants near me.",
                    "Where can I eat {cuisine} food around here?",
                ]
            },
        ),
        Tool(
            "search_flights",
            "Search for available flights.",
            {
                "origin": Str("IATA code of the departure airport", pick(IATA)),
                "destination": Str(
                    "IATA code of the arrival airport", pick(IATA), inventable=False
                ),
                "date": date_param("Departure date in ISO 8601 format (YYYY-MM-DD)"),
                "passengers": Int(
                    "Number of passengers", int_range(2, 6), minimum=1, maximum=9, kind="int"
                ),
                "cabin": Str(
                    "Cabin class",
                    pick(["economy", "premium_economy", "business", "first"]),
                    enum=["economy", "premium_economy", "business", "first"],
                    display={
                        "economy": "economy class",
                        "premium_economy": "premium economy",
                        "business": "business class",
                        "first": "first class",
                    },
                    syn={
                        "economy": ["coach"],
                        "premium_economy": ["the extra-legroom section"],
                        "business": ["the business cabin"],
                        "first": ["the first-class cabin"],
                    },
                ),
            },
            ["origin", "destination", "date"],
            [
                "Find flights from {origin} to {destination} on {date}.",
                "Search for flights {origin} to {destination} leaving {date}.",
                "I need a flight from {origin} to {destination} on {date}.",
            ],
            {"passengers": [" for {passengers} passengers"], "cabin": [" in {cabin}"]},
            {
                "destination": ["Find flights from {origin} on {date}."],
                "date": ["Find flights from {origin} to {destination}."],
            },
        ),
        Tool(
            "book_hotel",
            "Book a hotel room.",
            {
                "hotel_name": Str("Name of the hotel", pick(HOTELS)),
                "check_in": date_param("Check-in date in ISO 8601 format (YYYY-MM-DD)"),
                "check_out": date_param("Check-out date in ISO 8601 format (YYYY-MM-DD)"),
                "guests": Int(
                    "Number of guests", int_range(2, 6), minimum=1, maximum=10, kind="int"
                ),
                "room_type": Str(
                    "Room type",
                    pick(["single", "double", "suite"]),
                    enum=["single", "double", "suite"],
                    display={
                        "single": "a single room",
                        "double": "a double room",
                        "suite": "a suite",
                    },
                ),
                "breakfast_included": Bool(
                    "Whether breakfast is included",
                    (
                        ["with breakfast included", "including breakfast"],
                        ["without breakfast", "no breakfast needed"],
                    ),
                ),
            },
            ["hotel_name", "check_in", "check_out", "guests"],
            [
                "Book {hotel_name} from {check_in} to {check_out} for {guests} guests.",
                "Reserve a room at {hotel_name}, checking in {check_in} and out {check_out}, for {guests} guests.",
            ],
            {"room_type": [", in {room_type}"], "breakfast_included": [", {breakfast_included}"]},
            {"check_in": ["Book {hotel_name} until {check_out} for {guests} guests."]},
        ),
        Tool(
            "get_stock_price",
            "Get the latest share price for a ticker symbol.",
            {
                "ticker": Str("Ticker symbol", pick(TICKERS), inventable=False),
                "currency": Str("Currency for the price", pick(CURRENCIES)),
            },
            ["ticker"],
            [
                "What's the current share price of {ticker}?",
                "Get me the stock price for {ticker}.",
                "How is {ticker} trading right now?",
            ],
            {"currency": [", quoted in {currency}"]},
            {"ticker": ["What's the current share price?", "Get me the latest stock price."]},
        ),
        Tool(
            "translate_text",
            "Translate text into another language.",
            {
                "text": Str("Text to translate", pick(PHRASES_EN), inventable=False),
                "target_language": Str("Language to translate into", pick(LANGUAGES)),
                "source_language": Str("Language of the original text", pick(["English"])),
            },
            ["text", "target_language"],
            [
                'Translate "{text}" into {target_language}.',
                'How do you say "{text}" in {target_language}?',
                "Please translate this to {target_language}: {text}",
            ],
            {"source_language": [" (the original is in {source_language})"]},
            {
                "text": [
                    "Translate this into {target_language}.",
                    "Can you put that into {target_language} for me?",
                ]
            },
            [
                "Translate the text between the dashed lines into {target_language} (do not include the dashes):"
                "\n---\n{text}\n---"
            ],
        ),
        Tool(
            "add_contact",
            "Add a person to the address book.",
            {
                "name": Str("Full name", s_person),
                "phone": Str("Phone number", s_phone),
                "email": Str("Email address", s_email),
                "address": Param(
                    "object",
                    "Postal address",
                    _s_address_obj,
                    properties={
                        "street": {"type": "string"},
                        "city": {"type": "string"},
                        "postal_code": {"type": "string"},
                        "country": {"type": "string"},
                    },
                    nested_required=["street", "city", "postal_code", "country"],
                ),
            },
            ["name", "phone"],
            [
                "Add {name} to my contacts with phone number {phone}.",
                "Save a new contact: {name}, {phone}.",
            ],
            {
                "email": [" and email {email}"],
                "address": [". Their address is {street}, {city} {postal_code}, {country}"],
            },
            variables=_vars_contact,
        ),
        Tool(
            "place_order",
            "Place an order for one or more products.",
            {
                "items": Param(
                    "array",
                    "Products to order",
                    _s_items,
                    items={
                        "type": "object",
                        "properties": {"sku": {"type": "string"}, "quantity": {"type": "integer"}},
                        "required": ["sku", "quantity"],
                    },
                ),
                "shipping": Param(
                    "object",
                    "Shipping details",
                    _s_shipping,
                    properties={
                        "method": {"type": "string", "enum": ["standard", "express", "overnight"]},
                        "address": {
                            "type": "object",
                            "properties": {
                                "street": {"type": "string"},
                                "city": {"type": "string"},
                                "postal_code": {"type": "string"},
                            },
                            "required": ["street", "city", "postal_code"],
                        },
                    },
                    nested_required=["method", "address"],
                ),
                "gift_wrap": Bool(
                    "Wrap the order as a gift",
                    (
                        ["gift wrapped", "wrapped as a gift"],
                        ["with no gift wrapping", "not gift wrapped"],
                    ),
                ),
            },
            ["items", "shipping"],
            [
                "Order {items_text} and ship it by {method} to {street}, {city} {postal_code}.",
                "I'd like to buy {items_text}, delivered by {method} shipping to {street}, {city} {postal_code}.",
            ],
            {"gift_wrap": [", {gift_wrap}"]},
            variables=_vars_order,
        ),
        Tool(
            "search_products",
            "Search the product catalog with optional filters.",
            {
                "query": Str("Search text", pick(PRODUCT_QUERIES)),
                "filters": Param(
                    "object",
                    "Optional filters",
                    _s_filters,
                    properties={
                        "category": {"type": "string"},
                        "max_price": {"type": "number"},
                        "min_rating": {"type": "number"},
                        "in_stock": {"type": "boolean"},
                    },
                    nested_required=[],
                ),
            },
            ["query"],
            ["Search the catalog for {query}.", "Find me a {query} in the store."],
            {"filters": [", {filters_text}"]},
            variables=_vars_filters,
        ),
        Tool(
            "play_music",
            "Play music by song, artist or playlist name.",
            {
                "query": Str("What to play", pick(ARTISTS), inventable=False),
                "shuffle": Bool(
                    "Shuffle the tracks",
                    (["on shuffle", "in shuffle mode"], ["in order", "without shuffling"]),
                ),
                "volume": Int(
                    "Volume from 0 to 100",
                    pick([20, 35, 50, 65, 80]),
                    minimum=0,
                    maximum=100,
                    kind="int",
                ),
            },
            ["query"],
            ["Play {query}.", "Put on some {query}, please."],
            {"shuffle": [", {shuffle}"], "volume": [" at volume {volume}"]},
            {"query": ["Play something for me.", "Put on some music, please."]},
        ),
        Tool(
            "create_task",
            "Create a task in the user's to-do list.",
            {
                "title": Str("Task title", pick(TASK_TITLES)),
                "priority": Str(
                    "Task priority",
                    pick(["low", "medium", "high", "urgent"]),
                    enum=["low", "medium", "high", "urgent"],
                    syn={
                        "low": ["not important", "can wait"],
                        "medium": ["normal", "average"],
                        "high": ["important"],
                        "urgent": ["ASAP", "critical"],
                    },
                ),
                "due_date": date_param("Due date in ISO 8601 format (YYYY-MM-DD)"),
                "tags": Arr("Tags to attach", pick_n(TAGS, 2, 3)),
                "assignee": Str("Person responsible", s_person),
            },
            ["title"],
            ['Create a task: "{title}".', 'Add "{title}" to my to-do list.'],
            {
                "priority": [", priority: {priority}"],
                "due_date": [", due {due_date}"],
                "tags": [", tagged {tags}"],
                "assignee": [", assigned to {assignee}"],
            },
        ),
        Tool(
            "set_timer",
            "Start a countdown timer.",
            {
                "minutes": Int(
                    "Duration in minutes", int_range(2, 90), minimum=1, kind="int", inventable=False
                ),
                "label": Str(
                    "Name for the timer", pick(["pasta", "laundry", "tea", "stretch break", "oven"])
                ),
            },
            ["minutes"],
            ["Set a timer for {minutes} minutes.", "Start a countdown of {minutes} minutes."],
            {"label": [" called {label}"]},
            {"minutes": ["Set a timer for me.", "Start a countdown timer."]},
        ),
        Tool(
            "create_note",
            "Create a note.",
            {
                "title": Str("Note title", pick(NOTE_TITLES)),
                "body": Str("Note text", pick(EMAIL_BODIES)),
                "tags": Arr("Tags to attach", pick_n(TAGS, 2, 3)),
            },
            ["title", "body"],
            [
                'Create a note titled "{title}" that says "{body}"',
                'Jot down a note called "{title}": "{body}"',
            ],
            {"tags": [", tagged {tags}"]},
            {},
            [
                'Create a note titled "{title}". The text must be exactly what is between the dashed lines '
                "(without the dashes):\n---\n{body}\n---"
            ],
        ),
        Tool(
            "create_issue",
            "Open an issue in a code repository.",
            {
                "repo": Str("Repository in owner/name form", s_repo, inventable=False),
                "title": Str("Issue title", pick(ISSUE_TITLES)),
                "body": Str("Issue description", pick(ISSUE_BODIES)),
                "labels": Arr("Labels to apply", pick_n(LABELS, 2, 3)),
                "assignee": Str("Username to assign", lambda r: r.choice(FIRST_NAMES).lower()),
                "milestone": Str("Milestone name", pick(MILESTONES)),
            },
            ["repo", "title", "body"],
            [
                'Open an issue in {repo} titled "{title}" with the description "{body}"',
                'File a bug in {repo} called "{title}". The description is "{body}"',
            ],
            {
                "labels": [", labeled {labels}"],
                "assignee": [", assigned to {assignee}"],
                "milestone": [", for milestone {milestone}"],
            },
            {"repo": ['Open an issue titled "{title}" with the description "{body}"']},
            [
                'Open an issue in {repo} titled "{title}". The description must be exactly the text between the '
                "dashed lines (without the dashes):\n---\n{body}\n---"
            ],
        ),
        Tool(
            "set_thermostat",
            "Set the target temperature of the home thermostat.",
            {
                "temperature": Num(
                    "Target temperature in degrees Celsius",
                    lambda r: float(r.randint(16, 26)),
                    inventable=False,
                ),
                "mode": Str(
                    "Operating mode",
                    pick(["heat", "cool", "auto"]),
                    enum=["heat", "cool", "auto", "off"],
                    syn={
                        "heat": ["heating", "warming"],
                        "cool": ["cooling", "air-conditioning"],
                        "auto": ["automatic"],
                    },
                ),
                "room": Str("Room to apply it to", pick(ROOMS)),
            },
            ["temperature"],
            [
                "Set the thermostat to {temperature} degrees.",
                "Make it {temperature} degrees in here.",
            ],
            {"mode": [", using {mode} mode"], "room": [" for the {room}"]},
            {
                "temperature": [
                    "Adjust the thermostat for me.",
                    "Change the thermostat setting, please.",
                ]
            },
        ),
        Tool(
            "book_ride",
            "Book a ride-share car.",
            {
                "pickup": Str("Pickup address", s_address, inventable=False),
                "dropoff": Str("Drop-off address", s_address, inventable=False),
                "ride_type": Str(
                    "Type of car",
                    pick(["economy", "comfort", "xl"]),
                    enum=["economy", "comfort", "xl"],
                ),
            },
            ["pickup", "dropoff"],
            ["Book a ride from {pickup} to {dropoff}.", "I need a car from {pickup} to {dropoff}."],
            {"ride_type": [", ride type {ride_type}"]},
            {"dropoff": ["Book a ride from {pickup}."]},
        ),
        Tool(
            "send_payment",
            "Send money to a recipient.",
            {
                "recipient": Str("Recipient's email address", s_email, inventable=False),
                "amount": Num("Amount to send", s_amount, inventable=False),
                "currency": Str("Three-letter ISO 4217 currency code", pick(CURRENCIES)),
                "memo": Str(
                    "Note attached to the payment",
                    pick(["rent share", "concert tickets", "lunch", "thank you"]),
                ),
            },
            ["recipient", "amount", "currency"],
            ["Send {amount} {currency} to {recipient}.", "Pay {recipient} {amount} {currency}."],
            {"memo": [' with the memo "{memo}"']},
            {
                "recipient": ["Send {amount} {currency} right away."],
                "amount": ["Pay {recipient} the usual amount in {currency}."],
            },
        ),
        Tool(
            "get_directions",
            "Get turn-by-turn directions between two places.",
            {
                "origin": Str("Starting address", s_address, inventable=False),
                "destination": Str("Destination address", s_address, inventable=False),
                "mode": Str(
                    "Travel mode",
                    pick(["driving", "walking", "cycling", "transit"]),
                    enum=["driving", "walking", "cycling", "transit"],
                    display={
                        "driving": "car",
                        "walking": "foot",
                        "cycling": "bike",
                        "transit": "public transit",
                    },
                ),
            },
            ["origin", "destination"],
            [
                "How do I get from {origin} to {destination}?",
                "Give me directions from {origin} to {destination}.",
            ],
            {"mode": [" by {mode}"]},
            {"destination": ["Give me directions from {origin}."]},
        ),
        Tool(
            "check_order_status",
            "Look up the status of an order.",
            {"order_id": Str("Order identifier such as ORD-12345", s_order_id, inventable=False)},
            ["order_id"],
            ["What's the status of order {order_id}?", "Where is my order {order_id}?"],
            {},
            {"order_id": ["What's the status of my order?", "Where is my order right now?"]},
        ),
        Tool(
            "cancel_order",
            "Cancel an order that has not shipped.",
            {
                "order_id": Str("Order identifier such as ORD-12345", s_order_id, inventable=False),
                "reason": Str(
                    "Why the order is cancelled",
                    pick(["ordered by mistake", "found a better price"]),
                ),
            },
            ["order_id"],
            ["Cancel order {order_id}.", "Please cancel {order_id}."],
            {"reason": [", reason: {reason}"]},
            {"order_id": ["Cancel my order."]},
        ),
        Tool(
            "list_orders",
            "List the recent orders of a customer.",
            {
                "customer_id": Str(
                    "Customer identifier such as C-1234",
                    lambda r: f"C-{r.randint(1000, 9999)}",
                    inventable=False,
                )
            },
            ["customer_id"],
            [
                "List the recent orders for customer {customer_id}.",
                "Show me the orders for {customer_id}.",
            ],
            {},
            {"customer_id": ["Show me the recent orders."]},
        ),
        Tool(
            "search_contacts",
            "Search the address book by name.",
            {"name": Str("Name or part of a name", pick(FIRST_NAMES), inventable=False)},
            ["name"],
            ["Look up {name} in my contacts.", "Find {name} in the address book."],
            {},
            {"name": ["Look someone up in my contacts."]},
        ),
        Tool(
            "get_news",
            "Fetch recent news headlines on a topic.",
            {
                "topic": Str("Topic to search for", pick(TOPICS)),
                "count": Int(
                    "Number of headlines", pick([3, 5, 7, 10]), minimum=1, maximum=20, kind="int"
                ),
                "language": Str(
                    "Language of the articles", pick(["English", "French", "Spanish", "German"])
                ),
            },
            ["topic"],
            ["Get the latest news about {topic}.", "What are the recent headlines on {topic}?"],
            {"count": [" (just {count} articles)"], "language": [" in {language}"]},
        ),
    ]
    by_name = {tool.name: tool for tool in tools}
    by_name["convert_units"].fixup = _fix_units
    by_name["book_hotel"].fixup = _fix_hotel_dates
    by_name["search_flights"].fixup = _fix_trip
    by_name["get_directions"].fixup = _distinct("origin", "destination")
    by_name["book_ride"].fixup = _distinct("pickup", "dropoff")
    by_name["convert_currency"].fixup = _fix_currencies
    return by_name


TOOLS: dict[str, Tool] = _build()


def tool(name: str) -> Tool:
    return TOOLS[name]
