"""
AstroNexus AI — Conversation History Test

Shows the difference history makes on follow-up questions.
Run: python -m backend.tests.test_conversation
"""
from __future__ import annotations

import sys
import logging

try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

logging.basicConfig(level=logging.WARNING)

CYAN  = "\033[96m"
GREEN = "\033[92m"
BOLD  = "\033[1m"
RESET = "\033[0m"
DIM   = "\033[2m"

# A 5-turn conversation where every follow-up relies on the previous answer
CONVERSATION = [
    "What is the main contribution of this paper?",
    "Explain that in simple terms, like I am a student.",
    "What datasets did they use to validate it?",
    "What are its main limitations?",
    "How could those limitations be addressed in future work?",
]


def run():
    from backend.agents.orchestrator import run as orchestrate

    print(f"\n{'═'*65}")
    print(f"{BOLD}  ASTRONEXUS — CONVERSATION HISTORY TEST{RESET}")
    print(f"  5 turns, each building on the previous answer")
    print(f"{'═'*65}")

    history = []

    for i, query in enumerate(CONVERSATION, 1):
        print(f"\n{'─'*65}")
        print(f"{BOLD}Turn {i}{RESET}  {CYAN}{query}{RESET}")

        if history:
            print(f"{DIM}  (history: {len(history)} previous turn(s) available){RESET}")

        try:
            result = orchestrate(
                query=                query,
                paper_loaded=         True,
                conversation_history= history,
            )

            answer  = result.get("final_answer", "")
            history = result.get("conversation_history", history)
            meta    = result.get("metadata") or {}
            evl     = meta.get("evaluation") or {}

            print(f"\n  {BOLD}Answer:{RESET}")
            words = answer.split()
            line  = "  "
            for word in words:
                if len(line) + len(word) > 67:
                    print(line)
                    line = "  " + word + " "
                else:
                    line += word + " "
            if line.strip():
                print(line)

            print(
                f"\n  {DIM}conf={evl.get('confidence','?')}  "
                f"grounding={evl.get('grounding_score',0):.2f}  "
                f"history_after={len(history)}{RESET}"
            )

        except Exception as e:
            print(f"  ERROR: {e}")
            break

    print(f"\n{'═'*65}")
    print(f"{GREEN}  Conversation complete — {len(history)} turns in history{RESET}")
    print(f"{'═'*65}\n")


if __name__ == "__main__":
    run()