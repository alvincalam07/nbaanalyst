"""
Wemby-GM: NBA Trade Analyst Agent
Model: claude-haiku-4-5-20251001  |  Local-first, SQLite session resumption
"""

# ─────────────────────────────────────────────────────────────────────────────
# 0.  IMPORTS
# ─────────────────────────────────────────────────────────────────────────────
import asyncio
import json
import os
import sqlite3
import sys
import uuid
from datetime import datetime
from typing import Any, Dict, List, Optional

# Force UTF-8 output on Windows (cp1252 cannot encode characters the LLM returns)
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

import anthropic
from dotenv import load_dotenv
from pydantic import BaseModel, Field, model_validator

# ─────────────────────────────────────────────────────────────────────────────
# 1.  ENVIRONMENT
# ─────────────────────────────────────────────────────────────────────────────
load_dotenv()
_API_KEY: str = os.getenv("ANTHROPIC_API_KEY", "")
if not _API_KEY:
    raise EnvironmentError("ANTHROPIC_API_KEY is missing — set it in .env")

# ─────────────────────────────────────────────────────────────────────────────
# 2.  MOCK DATA LAYER
# ─────────────────────────────────────────────────────────────────────────────
MOCK_PLAYERS: List[Dict[str, Any]] = [
    {"id": "p001", "name": "Victor Wembanyama",  "team": "SAS", "salary": 12_100_518,  "epm": 8.4, "position": "C"},
    {"id": "p002", "name": "Devin Booker",        "team": "PHX", "salary": 35_342_000,  "epm": 3.8, "position": "SG"},
    {"id": "p003", "name": "Bam Adebayo",          "team": "MIA", "salary": 32_600_000,  "epm": 3.1, "position": "C"},
    {"id": "p004", "name": "Pascal Siakam",        "team": "IND", "salary": 37_893_408,  "epm": 2.9, "position": "PF"},
    {"id": "p005", "name": "Tyrese Haliburton",    "team": "IND", "salary": 22_281_000,  "epm": 4.5, "position": "PG"},
    {"id": "p006", "name": "Zach LaVine",          "team": "CHI", "salary": 43_860_600,  "epm": 1.2, "position": "SG"},
    {"id": "p007", "name": "Scottie Barnes",       "team": "TOR", "salary": 8_041_800,   "epm": 2.7, "position": "SF"},
    {"id": "p008", "name": "Brandon Ingram",       "team": "NOP", "salary": 33_833_400,  "epm": 2.1, "position": "SF"},
    {"id": "p009", "name": "Karl-Anthony Towns",   "team": "MIN", "salary": 49_279_972,  "epm": 2.4, "position": "C"},
    {"id": "p010", "name": "DeMar DeRozan",        "team": "SAC", "salary": 28_600_000,  "epm": 1.8, "position": "SF"},
    {"id": "p011", "name": "Josh Giddey",          "team": "CHI", "salary": 6_630_600,   "epm": 0.8, "position": "PG"},
    {"id": "p012", "name": "Nassir Little",        "team": "SAS", "salary": 4_200_000,   "epm": 0.3, "position": "SF"},
    {"id": "p013", "name": "Keldon Johnson",       "team": "SAS", "salary": 14_350_000,  "epm": 1.1, "position": "SF"},
    {"id": "p014", "name": "Obi Toppin",           "team": "IND", "salary": 13_000_000,  "epm": 0.9, "position": "PF"},
    {"id": "p015", "name": "Andrew Nembhard",      "team": "IND", "salary": 3_800_000,   "epm": 1.4, "position": "PG"},
]

TEAM_CAP_DATA: Dict[str, Dict[str, Any]] = {
    "SAS": {"cap_space": 45_000_000, "hard_cap": 185_000_000, "current_payroll":  95_000_000},
    "PHX": {"cap_space":  5_000_000, "hard_cap": 185_000_000, "current_payroll": 168_000_000},
    "MIA": {"cap_space":  8_000_000, "hard_cap": 185_000_000, "current_payroll": 165_000_000},
    "IND": {"cap_space": 12_000_000, "hard_cap": 185_000_000, "current_payroll": 142_000_000},
    "CHI": {"cap_space":  2_000_000, "hard_cap": 185_000_000, "current_payroll": 171_000_000},
    "TOR": {"cap_space": 28_000_000, "hard_cap": 185_000_000, "current_payroll": 122_000_000},
    "NOP": {"cap_space": 18_000_000, "hard_cap": 185_000_000, "current_payroll": 138_000_000},
    "MIN": {"cap_space":  1_000_000, "hard_cap": 185_000_000, "current_payroll": 178_000_000},
    "SAC": {"cap_space": 10_000_000, "hard_cap": 185_000_000, "current_payroll": 155_000_000},
}

CBA_SALARY_MATCH_LIMIT: float = 1.25

# ─────────────────────────────────────────────────────────────────────────────
# 3.  PYDANTIC v2 SCHEMAS
# ─────────────────────────────────────────────────────────────────────────────

class Player(BaseModel):
    id: str
    name: str
    team: str
    salary: int = Field(..., ge=0)
    epm: float
    position: str

    @model_validator(mode="after")
    def validate_position(self) -> "Player":
        valid_positions = {"PG", "SG", "SF", "PF", "C"}
        if self.position not in valid_positions:
            raise ValueError(
                f"Invalid position '{self.position}' for {self.name}. "
                f"Must be one of {valid_positions}."
            )
        return self


class TradeProposal(BaseModel):
    game_id: str
    team_a: str
    team_b: str
    team_a_sends: List[Player]
    team_b_sends: List[Player]
    team_a_outgoing_salary: int = 0
    team_b_outgoing_salary: int = 0
    salary_match_ratio: float = 0.0
    epm_delta_team_a: float = 0.0
    epm_delta_team_b: float = 0.0

    @model_validator(mode="after")
    def compute_metrics(self) -> "TradeProposal":
        self.team_a_outgoing_salary = sum(p.salary for p in self.team_a_sends)
        self.team_b_outgoing_salary = sum(p.salary for p in self.team_b_sends)
        if self.team_b_outgoing_salary > 0:
            self.salary_match_ratio = (
                self.team_a_outgoing_salary / self.team_b_outgoing_salary
            )
        # EPM delta = EPM gained minus EPM lost for each team
        self.epm_delta_team_a = (
            sum(p.epm for p in self.team_b_sends)
            - sum(p.epm for p in self.team_a_sends)
        )
        self.epm_delta_team_b = (
            sum(p.epm for p in self.team_a_sends)
            - sum(p.epm for p in self.team_b_sends)
        )
        return self


class CBARuleViolation(BaseModel):
    violation_type: str = "SALARY_MATCH_VIOLATION"
    message: str
    team_a_salary_sent: int
    team_b_salary_sent: int
    actual_ratio: float
    max_allowed_ratio: float = CBA_SALARY_MATCH_LIMIT
    self_correction_attempt: int = 0
    timestamp: str = Field(default_factory=lambda: datetime.utcnow().isoformat())
    offending_player_ids: List[str] = Field(default_factory=list)


# ─────────────────────────────────────────────────────────────────────────────
# 4.  SQLITE SESSION MANAGER
# ─────────────────────────────────────────────────────────────────────────────
DB_PATH = "wemby_gm.db"


def _init_db() -> sqlite3.Connection:
    conn = sqlite3.connect(DB_PATH)
    conn.execute("""
        CREATE TABLE IF NOT EXISTS sessions (
            session_id  TEXT    NOT NULL,
            step        INTEGER NOT NULL,
            history     TEXT    NOT NULL,
            updated_at  TEXT    NOT NULL,
            PRIMARY KEY (session_id, step)
        )
    """)
    conn.commit()
    return conn


def save_session_state(session_id: str, step: int, history_json: str) -> None:
    conn = _init_db()
    try:
        with conn:
            conn.execute(
                "INSERT OR REPLACE INTO sessions "
                "(session_id, step, history, updated_at) VALUES (?,?,?,?)",
                (session_id, step, history_json, datetime.utcnow().isoformat()),
            )
    finally:
        conn.close()


def load_session_state(session_id: str) -> Optional[Dict[str, Any]]:
    conn = _init_db()
    try:
        row = conn.execute(
            "SELECT step, history FROM sessions "
            "WHERE session_id=? ORDER BY step DESC LIMIT 1",
            (session_id,),
        ).fetchone()
    finally:
        conn.close()
    if row is None:
        return None
    return {"step": row[0], "history": json.loads(row[1])}


# ─────────────────────────────────────────────────────────────────────────────
# 5.  CONTEXT TRIMMING + PROVENANCE
# ─────────────────────────────────────────────────────────────────────────────
MAX_CONTEXT_CHARS: int = 14_000  # ~3.5 k tokens at 4 chars/token


def trim_context(
    messages: List[Dict],
    game_id: str,
    max_chars: int = MAX_CONTEXT_CHARS,
) -> List[Dict]:
    """
    Slices the oldest messages from history when total char length exceeds
    max_chars. Always prepends a provenance tag carrying the game_id so
    downstream consumers can reconstruct lineage even after trimming.
    """
    provenance_tag: Dict[str, str] = {
        "role": "user",
        "content": (
            f"[PROVENANCE] game_id={game_id} "
            f"| context_trim_applied=true "
            f"| ts={datetime.utcnow().isoformat()}"
        ),
    }

    total_chars = sum(len(json.dumps(m)) for m in messages)
    if total_chars <= max_chars:
        return messages

    trimmed = messages[:]
    while (
        len(trimmed) > 4
        and sum(len(json.dumps(m)) for m in trimmed) > max_chars
    ):
        trimmed.pop(0)

    return [provenance_tag] + trimmed


# ─────────────────────────────────────────────────────────────────────────────
# 6.  CBA COMPLIANCE VALIDATION
# ─────────────────────────────────────────────────────────────────────────────

def check_cba_compliance(proposal: TradeProposal) -> None:
    """Raises ValueError(CBARuleViolation JSON) on salary-match breach."""
    a_out = proposal.team_a_outgoing_salary
    b_out = proposal.team_b_outgoing_salary
    if a_out == 0 or b_out == 0:
        return

    ratio_fwd = a_out / b_out
    ratio_rev = b_out / a_out
    worst_ratio = max(ratio_fwd, ratio_rev)

    if worst_ratio > CBA_SALARY_MATCH_LIMIT:
        violation = CBARuleViolation(
            message=(
                f"CBA salary-match violation: ratio {worst_ratio:.3f} exceeds "
                f"the {CBA_SALARY_MATCH_LIMIT * 100:.0f}% limit. "
                f"{proposal.team_a} sends ${a_out:,} | "
                f"{proposal.team_b} sends ${b_out:,}. "
                f"Rebalance player selection to close the salary gap."
            ),
            team_a_salary_sent=a_out,
            team_b_salary_sent=b_out,
            actual_ratio=worst_ratio,
            offending_player_ids=(
                [p.id for p in proposal.team_a_sends]
                + [p.id for p in proposal.team_b_sends]
            ),
        )
        raise ValueError(violation.model_dump_json(indent=2))


# ─────────────────────────────────────────────────────────────────────────────
# 7.  CONFIDENCE CALIBRATION
# ─────────────────────────────────────────────────────────────────────────────

def compute_confidence(
    proposal: TradeProposal,
    correction_applied: bool,
    critic_approved: bool,
) -> Dict[str, Any]:
    """
    Produces a [0.0, 1.0] confidence score with explicit calibration formula:
      50 % weight  — salary fairness (penalty for large ratio deviation from 1.0)
      40 % weight  — EPM net-gain for the better-off team (normalised over 5.0)
      −15 % penalty — self-correction was required
      +5 % bonus   — Critic stage approved the trade
    """
    salary_fairness = max(0.0, 1.0 - abs(proposal.salary_match_ratio - 1.0))

    best_epm_delta = max(proposal.epm_delta_team_a, proposal.epm_delta_team_b)
    epm_gain = min(1.0, max(0.0, best_epm_delta / 5.0))

    correction_penalty = 0.15 if correction_applied else 0.0
    critic_bonus = 0.05 if critic_approved else 0.0

    raw = (salary_fairness * 0.50) + (epm_gain * 0.40) - correction_penalty + critic_bonus
    score = round(min(1.0, max(0.0, raw)), 4)

    calibration_logic = (
        f"salary_fairness={salary_fairness:.4f}×0.50"
        f" + epm_gain={epm_gain:.4f}×0.40"
        f" - correction_penalty={correction_penalty:.2f}"
        f" + critic_bonus={critic_bonus:.2f}"
        f" = raw {raw:.4f} → clamped_score={score}"
    )

    return {"confidence_score": score, "calibration_logic": calibration_logic}


# ─────────────────────────────────────────────────────────────────────────────
# 8.  HELPERS
# ─────────────────────────────────────────────────────────────────────────────

def _extract_json(text: str) -> str:
    """Strip markdown fences and extract the outermost JSON object."""
    text = text.strip()
    if text.startswith("```"):
        newline = text.find("\n")
        if newline != -1:
            text = text[newline + 1:]
        closing = text.rfind("```")
        if closing != -1:
            text = text[:closing]
        text = text.strip()
    start = text.find("{")
    end = text.rfind("}") + 1
    if start != -1 and end > start:
        return text[start:end]
    return text


# ─────────────────────────────────────────────────────────────────────────────
# 9.  COORDINATOR / SUBAGENT ARCHITECTURE
# ─────────────────────────────────────────────────────────────────────────────
MODEL = "claude-haiku-4-5-20251001"

# Per-agent system prompts — each subagent knows only its own role.
_GENERATOR_SYSTEM = """\
You are GeneratorAgent for Wemby-GM, an NBA trade proposal drafter.
Given two team rosters and salary cap data, draft a valid two-team trade
that respects the CBA salary-match limit provided in the context.
Rules:
- Use ONLY player IDs, names, salaries, and positions from the supplied rosters.
- Never invent players, teams, or salaries.
- Return raw minified JSON only — no markdown fences, no commentary.
Schema: {"game_id":"...","team_a":"...","team_b":"...","team_a_sends":[<players>],"team_b_sends":[<players>]}
"""

_CRITIC_SYSTEM = """\
You are CriticAgent for Wemby-GM, an NBA trade compliance auditor.
Given a trade proposal JSON, evaluate it on three dimensions:
  1. CBA salary-match ratio (must be <= 1.25x in both directions)
  2. EPM delta gained/lost by each team
  3. Hard-cap headroom remaining after the trade
Return raw minified JSON only — no markdown, no commentary.
Schema: {"cba_compliant":true/false,"approved":true/false,"critique":"<one sentence>"}
"""

_REFINER_SYSTEM = """\
You are RefinerAgent for Wemby-GM, an NBA trade report writer.
Given a validated trade proposal, critic feedback, and pre-computed confidence
metrics, produce the final polished trade summary payload.
Include: full trade details, one-sentence rationale per team, and embed the
confidence_score and calibration_logic values verbatim as provided.
Return a single minified JSON object only — no markdown, no commentary.
"""


# ── SubAgent ──────────────────────────────────────────────────────────────────

class SubAgent:
    """
    Isolated single-purpose agent with its own message context.
    Each instance is independent — it sees only what the Coordinator
    explicitly passes to it, never the other agents' histories.
    """

    def __init__(self, name: str, system_prompt: str) -> None:
        self.name = name
        self.system_prompt = system_prompt
        self.messages: List[Dict] = []

    async def send(
        self,
        client: anthropic.AsyncAnthropic,
        user_content: str,
        max_tokens: int = 2048,
    ) -> str:
        self.messages.append({"role": "user", "content": user_content})
        response = await client.messages.create(
            model=MODEL,
            max_tokens=max_tokens,
            system=self.system_prompt,
            messages=self.messages,
        )
        reply = response.content[0].text
        self.messages.append({"role": "assistant", "content": reply})
        return reply


# ── CoordinatorAgent ──────────────────────────────────────────────────────────

class CoordinatorAgent:
    """
    Owns the end-to-end trade workflow. Spawns three SubAgents (Generator,
    Critic, Refiner), routes structured outputs between them, handles CBA
    validation and self-correction, and persists session checkpoints.

    Each SubAgent receives only the context it needs:
      Generator  — full roster + cap data
      Critic     — trade proposal JSON only
      Refiner    — proposal JSON + critic feedback + confidence metrics
    """

    def __init__(
        self,
        team_a: str,
        team_b: str,
        session_id: str,
        game_id: str,
        client: anthropic.AsyncAnthropic,
    ) -> None:
        self.team_a = team_a
        self.team_b = team_b
        self.session_id = session_id
        self.game_id = game_id
        self.client = client

        self.generator = SubAgent("generator", _GENERATOR_SYSTEM)
        self.critic    = SubAgent("critic",    _CRITIC_SYSTEM)
        self.refiner   = SubAgent("refiner",   _REFINER_SYSTEM)

    # ── internal helpers ─────────────────────────────────────────────────────

    def _build_context(self) -> str:
        return json.dumps({
            "game_id": self.game_id,
            "team_a": self.team_a,
            "team_b": self.team_b,
            "roster_a": [p for p in MOCK_PLAYERS if p["team"] == self.team_a],
            "roster_b": [p for p in MOCK_PLAYERS if p["team"] == self.team_b],
            "cap_a": TEAM_CAP_DATA.get(self.team_a, {}),
            "cap_b": TEAM_CAP_DATA.get(self.team_b, {}),
            "cba_salary_match_limit": CBA_SALARY_MATCH_LIMIT,
        }, indent=2)

    def _checkpoint(self, step: int) -> None:
        state = {
            "generator": self.generator.messages,
            "critic":    self.critic.messages,
            "refiner":   self.refiner.messages,
        }
        save_session_state(self.session_id, step, json.dumps(state))

    def _header(self, agent_name: str) -> None:
        print(f"\n{'='*64}")
        print(f"  [{agent_name.upper()}]   game_id={self.game_id}")
        print(f"{'='*64}")

    def _rejection_payload(self, exc: ValueError) -> Dict[str, Any]:
        print(f"\n[COORDINATOR] Both trade attempts failed CBA check — rejecting.")
        detail = json.loads(str(exc))
        print(f"  Reason: {detail['message']}")
        return {
            "game_id": self.game_id,
            "session_id": self.session_id,
            "trade_status": "rejected",
            "reason": "CBA salary-match violation persisted after self-correction",
            "detail": detail,
            "_meta": {
                "game_id": self.game_id,
                "session_id": self.session_id,
                "model": MODEL,
                "correction_applied": True,
                "critic_approved": False,
                "confidence_score": 0.0,
                "calibration_logic": "trade rejected — no CBA-compliant package possible",
            },
        }

    # ── orchestration ────────────────────────────────────────────────────────

    async def run(self) -> Dict[str, Any]:
        context = self._build_context()
        correction_applied = False
        critic_approved = False
        proposal: Optional[TradeProposal] = None

        # ── GENERATOR SubAgent ────────────────────────────────────────────────
        self._header("GeneratorAgent")
        gen_prompt = (
            f"Draft a two-team trade between {self.team_a} and {self.team_b}.\n"
            f"CBA salary-match limit: {CBA_SALARY_MATCH_LIMIT}x\n\n"
            f"Context:\n{context}"
        )
        try:
            gen_raw = await self.generator.send(self.client, gen_prompt)
            print(f"[GeneratorAgent]\n{gen_raw}\n")
            self._checkpoint(1)

            gen_data = json.loads(_extract_json(gen_raw))
            gen_data["game_id"] = self.game_id
            proposal = TradeProposal(**gen_data)
            check_cba_compliance(proposal)

        except ValueError as exc:
            # Coordinator intercepts CBA violation, routes correction back to
            # the Generator SubAgent (keeping the error in its own context).
            print(f"\n[COORDINATOR] CBA violation detected — dispatching self-correction to GeneratorAgent.")
            correction_applied = True

            correction_prompt = (
                f"CBA RULE VIOLATION — self-correct once.\n\n"
                f"Error (CBARuleViolation JSON):\n{exc}\n\n"
                f"Redraft so both salary packages are within "
                f"{CBA_SALARY_MATCH_LIMIT * 100:.0f}% of each other. "
                f"Use the same roster. Return minified JSON only."
            )
            corrected_raw = await self.generator.send(self.client, correction_prompt)
            print(f"[GeneratorAgent — corrected]\n{corrected_raw}\n")
            self._checkpoint(2)

            corrected_data = json.loads(_extract_json(corrected_raw))
            corrected_data["game_id"] = self.game_id
            proposal = TradeProposal(**corrected_data)
            try:
                check_cba_compliance(proposal)
            except ValueError as second_exc:
                return self._rejection_payload(second_exc)

        # ── CRITIC SubAgent ───────────────────────────────────────────────────
        # Critic receives only the validated trade JSON — not the generator history.
        self._header("CriticAgent")
        critic_prompt = (
            f"Audit this trade proposal for CBA compliance and basketball merit:\n\n"
            f"{json.dumps(proposal.model_dump(), indent=2)}"
        )
        critic_raw = await self.critic.send(self.client, critic_prompt)
        print(f"[CriticAgent]\n{critic_raw}\n")
        self._checkpoint(3)

        try:
            critic_approved = bool(json.loads(_extract_json(critic_raw)).get("approved", False))
        except (json.JSONDecodeError, KeyError):
            critic_approved = False

        # ── REFINER SubAgent ──────────────────────────────────────────────────
        # Refiner receives trade + critique + confidence — not generator history.
        self._header("RefinerAgent")
        confidence = compute_confidence(proposal, correction_applied, critic_approved)
        refiner_prompt = (
            f"Build the final trade summary payload.\n\n"
            f"Trade Proposal:\n{json.dumps(proposal.model_dump(), indent=2)}\n\n"
            f"Critic Feedback:\n{critic_raw}\n\n"
            f"Embed these confidence metrics verbatim:\n"
            f'  "confidence_score": {confidence["confidence_score"]},\n'
            f'  "calibration_logic": "{confidence["calibration_logic"]}"\n\n'
            f"Return a single minified JSON object. No markdown."
        )
        refiner_raw = await self.refiner.send(self.client, refiner_prompt)
        print(f"[RefinerAgent]\n{refiner_raw}\n")
        self._checkpoint(4)

        # ── Coordinator assembles final payload ───────────────────────────────
        try:
            final_payload: Dict[str, Any] = json.loads(_extract_json(refiner_raw))
        except json.JSONDecodeError:
            final_payload = {"raw_refiner_output": refiner_raw}

        final_payload["_meta"] = {
            "game_id": self.game_id,
            "session_id": self.session_id,
            "model": MODEL,
            "agents": ["GeneratorAgent", "CriticAgent", "RefinerAgent"],
            "correction_applied": correction_applied,
            "critic_approved": critic_approved,
            **confidence,
        }

        print(f"\n{'='*64}")
        print("  FINAL PAYLOAD")
        print(f"{'='*64}")
        print(json.dumps(final_payload, indent=2))
        return final_payload


# ── Public entry-point wrapper ────────────────────────────────────────────────

async def run_trade_agent(
    team_a: str,
    team_b: str,
    session_id: Optional[str] = None,
) -> Dict[str, Any]:
    """Thin wrapper: creates a CoordinatorAgent and runs it."""
    session_id = session_id or str(uuid.uuid4())
    game_id = f"GM-{session_id[:8].upper()}"
    client = anthropic.AsyncAnthropic(api_key=_API_KEY)

    prior = load_session_state(session_id)
    if prior:
        print(f"[COORDINATOR] Resuming session {session_id} from step {prior['step']}")

    coordinator = CoordinatorAgent(
        team_a=team_a,
        team_b=team_b,
        session_id=session_id,
        game_id=game_id,
        client=client,
    )
    return await coordinator.run()


# ─────────────────────────────────────────────────────────────────────────────
# 10.  ENTRY POINT
# ─────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    import argparse

    VALID_TEAMS = sorted(TEAM_CAP_DATA.keys())

    parser = argparse.ArgumentParser(
        prog="app.py",
        description="Wemby-GM: NBA trade analyst powered by claude-haiku-4-5-20251001",
    )
    parser.add_argument(
        "team_a",
        nargs="?",
        default="SAS",
        metavar="TEAM_A",
        help=f"Initiating team. Valid: {', '.join(VALID_TEAMS)} (default: SAS)",
    )
    parser.add_argument(
        "team_b",
        nargs="?",
        default="IND",
        metavar="TEAM_B",
        help=f"Receiving team. Valid: {', '.join(VALID_TEAMS)} (default: IND)",
    )
    parser.add_argument(
        "--session",
        default=None,
        metavar="SESSION_ID",
        help="Resume a prior session by its ID",
    )

    args = parser.parse_args()

    # Validate team codes
    errors = []
    if args.team_a.upper() not in TEAM_CAP_DATA:
        errors.append(f"Unknown team '{args.team_a}'. Valid teams: {', '.join(VALID_TEAMS)}")
    if args.team_b.upper() not in TEAM_CAP_DATA:
        errors.append(f"Unknown team '{args.team_b}'. Valid teams: {', '.join(VALID_TEAMS)}")
    if args.team_a.upper() == args.team_b.upper():
        errors.append("TEAM_A and TEAM_B must be different teams.")
    if errors:
        for e in errors:
            print(f"[ERROR] {e}")
        sys.exit(1)

    result = asyncio.run(
        run_trade_agent(
            team_a=args.team_a.upper(),
            team_b=args.team_b.upper(),
            session_id=args.session,
        )
    )
    print("\n[DONE] Wemby-GM trade analysis complete.")
    print(
        f"  confidence_score   : {result['_meta']['confidence_score']}\n"
        f"  calibration_logic  : {result['_meta']['calibration_logic']}\n"
        f"  correction_applied : {result['_meta']['correction_applied']}\n"
        f"  session_id         : {result['_meta']['session_id']}"
    )


# ─────────────────────────────────────────────────────────────────────────────
# 11.  TESTS  (pytest-discoverable — run: pytest app.py -v)
# ─────────────────────────────────────────────────────────────────────────────

def test_player_valid():
    p = Player(id="p001", name="Test", team="SAS", salary=5_000_000, epm=3.0, position="C")
    assert p.salary == 5_000_000
    assert p.position == "C"


def test_player_invalid_position():
    import pytest
    with pytest.raises(Exception):
        Player(id="p002", name="Bad", team="SAS", salary=5_000_000, epm=3.0, position="XY")


def test_trade_proposal_metrics():
    import pytest
    p1 = Player(id="p001", name="A", team="SAS", salary=20_000_000, epm=3.0, position="C")
    p2 = Player(id="p002", name="B", team="IND", salary=20_000_000, epm=5.0, position="PF")
    trade = TradeProposal(
        game_id="test-01",
        team_a="SAS",
        team_b="IND",
        team_a_sends=[p1],
        team_b_sends=[p2],
    )
    assert trade.salary_match_ratio == pytest.approx(1.0)
    # SAS sends epm=3.0, receives epm=5.0 → delta = +2.0
    assert trade.epm_delta_team_a == pytest.approx(2.0)
    # IND sends epm=5.0, receives epm=3.0 → delta = -2.0
    assert trade.epm_delta_team_b == pytest.approx(-2.0)


def test_cba_violation_raised():
    import pytest
    expensive = Player(id="p001", name="A", team="SAS", salary=50_000_000, epm=3.0, position="C")
    cheap = Player(id="p002", name="B", team="IND", salary=10_000_000, epm=2.0, position="PF")
    trade = TradeProposal(
        game_id="test-02",
        team_a="SAS",
        team_b="IND",
        team_a_sends=[expensive],
        team_b_sends=[cheap],
    )
    with pytest.raises(ValueError) as exc_info:
        check_cba_compliance(trade)
    error_body = json.loads(str(exc_info.value))
    assert error_body["violation_type"] == "SALARY_MATCH_VIOLATION"
    assert error_body["actual_ratio"] == pytest.approx(5.0)


def test_cba_compliant_trade_passes():
    p1 = Player(id="p001", name="A", team="SAS", salary=20_000_000, epm=3.0, position="C")
    p2 = Player(id="p002", name="B", team="IND", salary=20_000_000, epm=2.0, position="PF")
    trade = TradeProposal(
        game_id="test-03",
        team_a="SAS",
        team_b="IND",
        team_a_sends=[p1],
        team_b_sends=[p2],
    )
    check_cba_compliance(trade)  # Must not raise


def test_trim_context_trims_and_tags():
    large_messages = [{"role": "user", "content": "x" * 800}] * 30
    trimmed = trim_context(large_messages, game_id="GM-TEST01", max_chars=5_000)
    assert len(trimmed) < 30
    assert "PROVENANCE" in trimmed[0]["content"]
    assert "GM-TEST01" in trimmed[0]["content"]


def test_trim_context_no_trim_needed():
    small_messages = [{"role": "user", "content": "hello"}]
    result = trim_context(small_messages, game_id="GM-XYZ", max_chars=10_000)
    assert result == small_messages


def test_confidence_score_range():
    import pytest
    p1 = Player(id="p001", name="A", team="SAS", salary=20_000_000, epm=3.0, position="C")
    p2 = Player(id="p002", name="B", team="IND", salary=20_000_000, epm=4.0, position="PF")
    proposal = TradeProposal(
        game_id="test-04",
        team_a="SAS",
        team_b="IND",
        team_a_sends=[p1],
        team_b_sends=[p2],
    )
    conf = compute_confidence(proposal, correction_applied=False, critic_approved=True)
    assert 0.0 <= conf["confidence_score"] <= 1.0
    assert "calibration_logic" in conf
    assert "clamped_score" in conf["calibration_logic"]


def test_confidence_penalises_correction():
    p1 = Player(id="p001", name="A", team="SAS", salary=20_000_000, epm=3.0, position="C")
    p2 = Player(id="p002", name="B", team="IND", salary=20_000_000, epm=4.0, position="PF")
    proposal = TradeProposal(
        game_id="test-05",
        team_a="SAS",
        team_b="IND",
        team_a_sends=[p1],
        team_b_sends=[p2],
    )
    no_correction = compute_confidence(proposal, correction_applied=False, critic_approved=False)
    with_correction = compute_confidence(proposal, correction_applied=True, critic_approved=False)
    assert no_correction["confidence_score"] > with_correction["confidence_score"]


def test_session_save_and_load():
    global DB_PATH
    original_db_path = DB_PATH
    test_db = "test_wemby_gm_temp.db"
    DB_PATH = test_db
    try:
        history = [{"role": "user", "content": "draft a trade"}]
        save_session_state("sess-test-01", 1, json.dumps(history))
        result = load_session_state("sess-test-01")
        assert result is not None
        assert result["step"] == 1
        assert result["history"][0]["content"] == "draft a trade"
    finally:
        DB_PATH = original_db_path
        if os.path.exists(test_db):
            os.remove(test_db)


def test_session_missing_returns_none():
    global DB_PATH
    original_db_path = DB_PATH
    test_db = "test_wemby_gm_temp2.db"
    DB_PATH = test_db
    try:
        result = load_session_state("nonexistent-session-id")
        assert result is None
    finally:
        DB_PATH = original_db_path
        if os.path.exists(test_db):
            os.remove(test_db)


def test_extract_json_strips_fences():
    fenced = '```json\n{"key": "value"}\n```'
    raw = _extract_json(fenced)
    parsed = json.loads(raw)
    assert parsed["key"] == "value"


def test_extract_json_passthrough():
    clean = '{"key": 42}'
    assert json.loads(_extract_json(clean))["key"] == 42


def test_cba_rule_violation_schema():
    v = CBARuleViolation(
        message="test violation",
        team_a_salary_sent=50_000_000,
        team_b_salary_sent=10_000_000,
        actual_ratio=5.0,
        offending_player_ids=["p001", "p002"],
    )
    dumped = json.loads(v.model_dump_json())
    assert dumped["violation_type"] == "SALARY_MATCH_VIOLATION"
    assert dumped["max_allowed_ratio"] == CBA_SALARY_MATCH_LIMIT
    assert "p001" in dumped["offending_player_ids"]


def test_second_cba_violation_detected():
    # Both directions violate the 125% rule — no valid package exists between these two players.
    # Verifies that check_cba_compliance raises on both the initial and corrected proposals,
    # confirming the graceful-rejection path in run_trade_agent would trigger.
    import pytest

    expensive = Player(id="p006", name="Zach LaVine",    team="CHI", salary=43_860_600, epm=1.2, position="SG")
    cheap     = Player(id="p011", name="Josh Giddey",    team="CHI", salary=6_630_600,  epm=0.8, position="PG")
    ingram    = Player(id="p008", name="Brandon Ingram", team="NOP", salary=33_833_400,  epm=2.1, position="SF")

    # First attempt: LaVine for Ingram — ratio 1.296 > 1.25
    trade_v1 = TradeProposal(game_id="t1", team_a="CHI", team_b="NOP",
                             team_a_sends=[expensive], team_b_sends=[ingram])
    with pytest.raises(ValueError) as exc1:
        check_cba_compliance(trade_v1)
    assert json.loads(str(exc1.value))["actual_ratio"] == pytest.approx(1.296, rel=1e-2)

    # Corrected attempt: Giddey for Ingram — ratio 5.10 > 1.25 (worse)
    trade_v2 = TradeProposal(game_id="t1", team_a="CHI", team_b="NOP",
                             team_a_sends=[cheap], team_b_sends=[ingram])
    with pytest.raises(ValueError) as exc2:
        check_cba_compliance(trade_v2)
    assert json.loads(str(exc2.value))["actual_ratio"] == pytest.approx(5.10, rel=1e-2)


def test_graceful_rejection_payload_shape():
    rejection = {
        "game_id": "GM-TEST",
        "session_id": "test-session",
        "trade_status": "rejected",
        "reason": "CBA salary-match violation persisted after self-correction",
        "detail": {"violation_type": "SALARY_MATCH_VIOLATION", "actual_ratio": 5.1},
        "_meta": {
            "game_id": "GM-TEST",
            "session_id": "test-session",
            "model": MODEL,
            "correction_applied": True,
            "critic_approved": False,
            "confidence_score": 0.0,
            "calibration_logic": "trade rejected — no CBA-compliant package possible",
        },
    }
    assert rejection["trade_status"] == "rejected"
    assert rejection["_meta"]["confidence_score"] == 0.0
    assert rejection["_meta"]["correction_applied"] is True
    assert rejection["detail"]["violation_type"] == "SALARY_MATCH_VIOLATION"


def test_sub_agent_init():
    agent = SubAgent("generator", _GENERATOR_SYSTEM)
    assert agent.name == "generator"
    assert agent.messages == []
    assert "GeneratorAgent" in agent.system_prompt


def test_sub_agent_isolated_contexts():
    # Three SubAgents must have entirely separate message histories.
    gen  = SubAgent("generator", _GENERATOR_SYSTEM)
    crit = SubAgent("critic",    _CRITIC_SYSTEM)
    ref  = SubAgent("refiner",   _REFINER_SYSTEM)
    gen.messages.append({"role": "user", "content": "draft a trade"})
    assert crit.messages == []
    assert ref.messages  == []


def test_coordinator_agent_init():
    import anthropic as _ant
    client = _ant.AsyncAnthropic(api_key="sk-ant-test-key")
    coord = CoordinatorAgent("SAS", "IND", "sess-01", "GM-SESS0001", client)
    assert coord.team_a == "SAS"
    assert coord.team_b == "IND"
    assert isinstance(coord.generator, SubAgent)
    assert isinstance(coord.critic,    SubAgent)
    assert isinstance(coord.refiner,   SubAgent)
    # All three subagents start with empty message histories
    assert coord.generator.messages == []
    assert coord.critic.messages    == []
    assert coord.refiner.messages   == []


def test_coordinator_builds_context():
    import anthropic as _ant
    client = _ant.AsyncAnthropic(api_key="sk-ant-test-key")
    coord = CoordinatorAgent("PHX", "MIA", "sess-02", "GM-SESS0002", client)
    ctx = json.loads(coord._build_context())
    assert ctx["team_a"] == "PHX"
    assert ctx["team_b"] == "MIA"
    assert any(p["team"] == "PHX" for p in ctx["roster_a"])
    assert any(p["team"] == "MIA" for p in ctx["roster_b"])
    assert ctx["cba_salary_match_limit"] == CBA_SALARY_MATCH_LIMIT


def test_meta_includes_agent_list():
    # Final _meta should name all three subagents.
    agents = ["GeneratorAgent", "CriticAgent", "RefinerAgent"]
    meta = {"agents": agents, "model": MODEL}
    assert "GeneratorAgent" in meta["agents"]
    assert "CriticAgent"    in meta["agents"]
    assert "RefinerAgent"   in meta["agents"]
    assert len(meta["agents"]) == 3
